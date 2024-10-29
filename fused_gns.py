import torch
import torch.nn as nn
import torch.nn.functional as F

import normgnorm
from gnstracking import LayerNorm, TrackedLayerNorm

# on a small network, this is about 2x slower than the original
# should get better with larger networks as GEMM eats up the
# overhead
class PEGradNormFusedLayerNorm(TrackedLayerNorm):
    def __init__(self, ndim, bias):
        super(PEGradNormFusedLayerNorm, self).__init__(ndim, bias)
        self.weight_upegsqnorm = nn.Parameter(torch.full((2,), torch.nan))
        if bias:
            self.bias_upegsqnorm = nn.Parameter(torch.full((2,), torch.nan))
        else:
            self.bias_upegsqnorm = None

    def forward(self, input):
        return normgnorm.PEGLayerNorm.apply(input, self.weight, self.bias,
                                 self.weight_upegsqnorm,
                                 self.bias_upegsqnorm,
                                 self.normalized_shape, self.eps)
DIM = 10

class LNShimTest:
    def __init__(self):
        input_l = nn.Linear(DIM, DIM) # input linear layer
        self.input_state = input_l.state_dict()
        input_ln = LayerNorm(DIM, True) # input layernorm layer
        self.input_ln_state = input_ln.state_dict()
    def __call__(self, dtype, LN, B):
        # reset input state
        input_state = self.input_state.copy()
        input_state = {k: v.to(dtype) for k, v in input_state.items()}
        input_ln_state = self.input_ln_state.copy()
        input_ln_state = {k: v.to(dtype) for k, v in input_ln_state.items()}

        input_l = nn.Linear(DIM, DIM).to(dtype).to(dtype)
        input_l.load_state_dict(input_state)
        input_l.cuda()
        auxinput_l = nn.Linear(DIM, DIM).to(dtype).to(dtype)
        auxinput_l.load_state_dict(input_state)
        auxinput_l.cuda()

        # create a layernorm layer
        layernorm = LayerNorm(DIM, True).to(dtype)
        layernorm.load_state_dict(input_ln_state)
        layernorm.cuda()
        # create a shim layernorm layer
        auxlayernorm = LN(DIM, True).to(dtype)
        auxlayernorm.weight.data = layernorm.weight.data
        auxlayernorm.bias.data = layernorm.bias.data
        auxlayernorm.cuda()
        # create some input
        input = torch.randn(B, 5, DIM, requires_grad=True).to(dtype).cuda()
        auxinput = input.clone().detach().requires_grad_(True).to(dtype).cuda()

        # check output
        output = layernorm(input_l(input))
        shimoutput = auxlayernorm(auxinput_l(auxinput))
        print((output.cuda() - shimoutput).abs().max())

        if dtype == torch.float16:
            rtol, atol = 1e-2, 1e-3 # float16 is less precise
        else:
            rtol, atol = 1e-5, 1e-6
        assert torch.allclose(output, shimoutput.to(output.device), rtol=rtol, atol=atol), f"{output=}, {shimoutput=}"

        # check grads
        g = torch.randn_like(output)
        output.backward(g)
        shimoutput.backward(g)
        err = torch.abs(layernorm.weight.grad - auxlayernorm.weight.grad).max()
        if dtype == torch.float16:
            rtol, atol = 1e-2, 1e-3 # float16 is less precise
        else:
            rtol, atol = 1e-4, 1e-6
        assert torch.allclose(layernorm.weight.grad, auxlayernorm.weight.grad, rtol=rtol, atol=atol), f"{layernorm.weight.grad=}, {auxlayernorm.weight.grad=} {err=}"
        err = torch.abs(layernorm.bias.grad - auxlayernorm.bias.grad).max()
        assert torch.allclose(layernorm.bias.grad, auxlayernorm.bias.grad, rtol=rtol, atol=atol), f"{layernorm.bias.grad=}, {auxlayernorm.bias.grad=} {err=}"
        if dtype == torch.float16:
            rtol, atol = 1e-2, 1e-2
        else:
            rtol, atol = 1e-5, 1e-5
        for a, b in zip(input_l.parameters(), auxinput_l.parameters()):
            assert torch.allclose(a.grad, b.grad, rtol=rtol, atol=atol), f"{a.grad=}, {b.grad=}, {(a.grad - b.grad).abs().max()}"

        if LN in [PEGradNormFusedLayerNorm]:
            # compute pegsqnorm manually
            weight_pegsqnorm = 0.
            bias_pegsqnorm = 0.
            for i in range(B):
                layernorm.weight.grad.zero_()
                layernorm.bias.grad.zero_()
                output = layernorm(input_l(input[[i]]))
                output.backward(g[[i]])
                weight_pegsqnorm += (layernorm.weight.grad**2).sum()
                bias_pegsqnorm += (layernorm.bias.grad**2).sum()
            weight_pegsqnorm /= B
            bias_pegsqnorm /= B
            weight_pegsqnorm *= B**2
            bias_pegsqnorm *= B**2
            if dtype == torch.float16:
                rtol, atol = 1e-3, 1e-3
            else:
                rtol, atol = 1e-5, 1e-8
            assert torch.allclose(auxlayernorm.weight_pegsqnorm(), weight_pegsqnorm, rtol=rtol, atol=atol), f"{auxlayernorm.weight_pegsqnorm()=}, {weight_pegsqnorm=}"
            assert torch.allclose(auxlayernorm.bias_pegsqnorm(), bias_pegsqnorm, rtol=rtol, atol=atol), f"{auxlayernorm.bias_pegsqnorm()=}, {bias_pegsqnorm=}"
        print(f"{LN} test passed")

if __name__ == "__main__":
    # test shim linear
    import torch.nn.functional as F
    import torch
    import torch.nn as nn
    from torch.autograd import gradcheck
    # set seed
    torch.manual_seed(0)

    def loss(x):
        # dummy loss used for grad checking below
        return torch.square(x).mean()

    # test layernorm shims
    test = LNShimTest()
    for LN in [PEGradNormFusedLayerNorm]:
        for dtype in [torch.float16, torch.float32]:
            for B in [2,3,4]:
                test(dtype, LN, B)

