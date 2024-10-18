"""
Module to do bookkeeping of gradient norms and per example gradient norms,
which may be used to compute gradient noise scale.
"""

import dataclasses
from typing import Any, Dict
import torch

def exists(val):
    # lucidrains^TM helper
    # MIT License
    # Copyright (c) 2020 Phil Wang
    return val is not None

@dataclasses.dataclass
class Measurement:
    """
    A class to store each individual measurement of the gradient norm and per
    example gradient norm. It stores the norm, the per example norm, and any
    corresponding metadata (e.g. step number, param name, module index).
    """
    big_batch_norm: float
    small_batch_norm: float
    big_batch_size: int
    small_batch_size: int
    metadata: Dict[str, Any] = None # optional

    def __str__(self):
        return f"norm: {self.big_batch_norm:.2f}, small_batch_norm: {self.small_batch_norm:.2f}, metadata: {self.metadata}"

    def msg(self, prefix=""):
        # dictionary, for passing to wandb etc
        d = {prefix+k:v for k,v in dataclasses.asdict(self).items()}
        # remove metadata, it's not necessarily a scalar
        d.pop(prefix+"metadata")
        return d

def shared_weights(module):
    # detect shared parameters
    all_params = [id(p) for p in module.only_parameters()]
    # detect repeated ids (this will pass because `.parameters()` doesn't return the same object twice)
    assert len(all_params) == len(list(set(all_params))), "Repeated parameter ids"
    param_names = [n for n, p in module.only_named_parameters()]
    shared_params = []
    for n, m in module.named_modules():
        if hasattr(m, "weight"):
            n = n + ".weight"
            if n not in param_names:
                #print(f"Shared weight: {n}")
                for n2, p in module.only_named_parameters():
                    if p is getattr(m, "weight"):
                        #print(f"  {n2}")
                        # this is a shared parameter, so it should be excluded from iteration
                        shared_params.append(p)
    return shared_params

def named_parameters_and_buffers(module, enforce_coverage=False):
    for n, p in module.named_parameters():
        b = None
        try:
            b = module._buffers[n+"_upegsqnorm"]
        except KeyError:
            pass
        if b is None:
            if enforce_coverage:
                raise ValueError(f"Buffer not found for parameter {n}")
            continue
        yield n, p, b

class MeasurementTracker:
    """
    A class to track the gradient norm and per example gradient norm for a
    model. The user must pass in a list of tuples, where each tuple is a
    parameter and its corresponding buffer. The user can also pass in a list of
    names and/or module indexes, which will be used to annotate the recorded
    measurement.
    args:
    - params_and_buffers: a list of tuples containing the parameters and
      their corresponding buffers
    - names: a list of strings
    - indexes: a list of integers
    - callback: a function to call after recording each measurement, callback
      will be passed the recorded Measurement object, and should return
      a Measurement object for storage. If callback is None, the recorded
      will just be stored as is.
        callback(Measurement) -> Measurement
    - enforce_coverage: if True, will raise an error if there are parameters
      with no corresponding buffer
    - shared_params: a list of parameters that are shared between modules, this
      will be marked in the metadata of Measurement objects
    """
    def __init__(self, params_and_buffers, names=None, indexes=None,
                 callback=None, scaler=None, shared_params=None):
        self.params_and_buffers = params_and_buffers
        self.names = names
        self.indexes = indexes # module indexes
        self.callback = callback
        self.step_count = 0
        self.measurements = []
        self.scaler = scaler
        self.shared_params = shared_params

    @staticmethod
    def from_model(model, callback=None, enforce_coverage=False, scaler=None):
        assert isinstance(model, TrackedModule), "model must be a TrackedModule"
        names, params_and_buffers, indexes = [], [], []
        module_name2index = {n: i for i, (n, m) in enumerate(model.named_modules())}
        for n, p, b in named_parameters_and_buffers(model, enforce_coverage=enforce_coverage):
            names.append(n)
            params_and_buffers.append((p, b))
            mn = ".".join(n.split(".")[:-1])
            indexes.append(module_name2index[mn])
        shared_params = shared_weights(model)
        return GNSStatsTracker(params_and_buffers, names=names,
                            indexes=indexes, callback=callback, scaler=scaler,
                            shared_params=shared_params)

    def step(self, batch_size=None, step=None):
        # iterate over buffers and parameters, recording the norms
        for i, (p, b) in enumerate(self.params_and_buffers):
            current_scale = 1.
            if self.scaler:
                # correction due to grad scaling
                current_scale = self.scaler.get_scale()
                assert abs(current_scale-1.) > 1e-6, f"GNSStatsTracker.step MUST be called before scaler.step, current scale: {current_scale=}"
                b[0] /= (current_scale**2) # squared norms are scaled by the square of the scale
            # this appears to not be required, the grads are already summed
            norm = p.grad.float().norm().item() / current_scale
            observed_batch_size = int(b[1].item())
            per_example_norm = torch.prod(b).sqrt().item()
            metadata = {}
            if self.names:
                metadata["name"] = self.names[i]
            if self.indexes:
                metadata["index"] = self.indexes[i]
            if step is not None:
                metadata["step"] = step
            else:
                metadata["step"] = self.step_count
            if self.shared_params:
                metadata["shared"] = any(p is sp for sp in self.shared_params)
            measurement = Measurement(
                big_batch_norm=norm,
                small_batch_norm=per_example_norm,
                big_batch_size=batch_size,
                small_batch_size=1,
                metadata=metadata
            )
            if self.callback:
                # flow here is ugly to deal with either having a single callback
                # or a list of callbacks
                if type(self.callback) not in (list, tuple):
                    callbacks = [self.callback]
                else:
                    callbacks = self.callback
                _measurement = None
                for c in callbacks:
                    m = c(measurement)
                    if m is not None:
                        assert measurement is None, "Only one callback can return anything"
                        _measurement = m
                if _measurement is not None:
                    self.measurements.append(_measurement)
            else:
                self.measurements.append(measurement)
        self.step_count += 1

def unbiased_stats(sgn, spegn, b):
    # eq A.2 p.17 of https://arxiv.org/abs/1812.06162
    # with B_small = 1
    # so equal to Bessel correction for sample variance
    # sgn is the squared gradient norm
    # spegn is the squared per example gradient norm
    # b is batch size
    gtg = (b * sgn - spegn) / (b - 1.)
    trsigma = (spegn - sgn) / (1. - (1. / b))
    return gtg, trsigma

# Running mean callback to print the GNS during training
class GNS_EMA:
    def __init__(self, alpha=0.95):
        self.alpha = alpha
        self.step = None
        self.measurements = []
        self.sgn = 0.
        self.spegn = 0.
        self.gtg = None
        self.trsigma = None
        self.ema_gtg = None
        self.ema_trsigma = None
        self.alpha_cumprod = 1.

    def __call__(self, measurement):
        if measurement.metadata['shared']:
            return # don't track shared parameters (e.g. lm_head and embed)
        # aggregate measurements for a step
        if self.step is None:
            self.step = measurement.metadata["step"]
        if self.step != measurement.metadata["step"]:
            # new step
            self.step = measurement.metadata["step"]
            # compute total gn and pegn for the step
            self.sgn, self.spegn = 0., 0.
            batch_sizes = []
            for m in self.measurements:
                self.sgn += m.big_batch_norm**2
                self.spegn += m.small_batch_norm**2
                batch_sizes.append(m.big_batch_size)
            assert len(set(batch_sizes)) == 1, f"batch sizes must be the same {set(batch_sizes)}"
            batch_size = batch_sizes[0]
            self.gtg, self.trsigma = unbiased_stats(self.sgn, self.spegn, batch_size)
            if self.ema_gtg is None:
                self.ema_gtg = self.gtg
                self.ema_trsigma = self.trsigma
            else:
                self.ema_gtg = self.alpha * self.ema_gtg + (1. - self.alpha) * self.gtg
                self.ema_trsigma = self.alpha * self.ema_trsigma + (1. - self.alpha) * self.trsigma
            self.alpha_cumprod *= self.alpha
            self.measurements = []
        self.measurements.append(measurement)

    def get_gns(self, eps=1e-6):
        if self.ema_gtg is None:
            return 0.
        return max(self.ema_trsigma, eps) / (max(self.ema_gtg, eps))

    def get_gtg(self):
        if self.gtg is None:
            return 0.
        return self.gtg / (1. - self.alpha_cumprod)

    def get_trsigma(self):
        if self.trsigma is None:
            return 0.
        return self.trsigma / (1. - self.alpha_cumprod)

    def get_stats(self, prefix=""):
        return {
            prefix+"gtg": self.get_gtg(),
            prefix+"trsigma": self.get_trsigma(),
            prefix+"gns": self.get_gns(),
            prefix+"sgn": self.sgn,
            prefix+"spegn": self.spegn
        }

class AccumulateMeasurements:
    def __init__(self, prefix):
        self.msg = {}
        self.step = None
        self.prefix = prefix
    def __call__(self, m):
        if self.step is None:
            self.step = m.metadata["step"]
        if self.step != m.metadata["step"]:
            self.msg = {} # reset whenever we see a new step
            self.step = m.metadata["step"]
        # make msg from measurement m
        self.msg[self.prefix+m.metadata["name"]+"/norm"] = m.big_batch_norm
        self.msg[self.prefix+m.metadata["name"]+"/pegn"] = m.small_batch_norm
        self.msg[self.prefix+m.metadata["name"]+"/batch_size"] = m.big_batch_size
    def get_msg(self):
        msg = self.msg
        self.msg = {}
        return msg
