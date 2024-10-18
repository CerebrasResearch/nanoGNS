from functools import partial

import torch
import torch.nn as nn
import torch.nn.functional as F

def exists(x):
    return x is not None

def safe_sum(x, dim, keepdim=False):
    dtype = x.dtype
    x = x.float()
    return x.sum(dim=dim, keepdim=keepdim).to(dtype)

def safe_sqnorm(x, dim, keepdim=False):
    dtype = x.dtype
    x = x.float()
    return (x**2).sum(dim=dim, keepdim=keepdim).to(dtype)

def stack_with_ones(x, dim):
    # stacks a tensor with ones along a given dim
    ones = torch.ones_like(x)
    return torch.stack([x, ones], dim=dim)


class ModuleWithBuffers(nn.Module):
    def register_marked_buffer(self, name, init_func, marker):
        self.register_buffer(name, init_func())
        buffer = getattr(self, name)
        setattr(buffer, marker, True)


    def register_upegsqnorm_buffer(self, name, marker="is_pegsqnorm"):
        buffer_name = f"{name}_upegsqnorm"
        self.register_marked_buffer(buffer_name,
                                    lambda: torch.zeros(2),
                                    marker)
        # Add a method to the module for this buffer
        setattr(self, f"{name}_pegsqnorm",
                partial(self._to_pegsqnorm, buffer_name))

    def _to_pegsqnorm(self, buffer_name):
        return torch.prod(getattr(self, buffer_name))

    def named_buffers_with_marker(self, marker):
        """Gather all buffers marked with is_custom_buffer attribute."""
        return {
            name: buffer for name, buffer in self.named_buffers()
            if getattr(buffer, marker, False)
        }

def zero_sqgradnorm_buffers(model):
    for module in model.modules():
        if hasattr(module, 'named_buffers_with_marker'):
            for name, buffer in module.named_buffers_with_marker('is_pegsqnorm').items():
                buffer.zero_()

############################## Linear ##############################

def pe_linear_grad_norm(a, g):
    """
    Compute the per-example gradient norms for a linear layer.
    args:
        a (Tensor): the input tensor
        g (Tensor): the gradient tensor
    """
    b = a.shape[0]
    if a.ndim == 2:
        # use trad per-example trick
        # s = torch.einsum('bi,bi,bj,bj->b', g, g, a, a)
        a = a*a
        g = g*g
        s = a.sum(1)*g.sum(1)
        bias_s = g.sum(1)
    elif a.ndim > 2:
        b = a.shape[0]
        dg, da = g.shape[-1], a.shape[-1]
        # ensure that this is 3D
        a = a.reshape(b, -1, da)
        g = g.reshape(b, -1, dg)
        dw = torch.bmm(a.transpose(1, 2), g)
        b = dw.shape[0]
        s = safe_sum((dw * dw).view(b, -1), 1)
        db = g.sum(1)
        bias_s = safe_sum((db * db), -1)
    return stack_with_ones(s, -1).sum(0), stack_with_ones(bias_s, -1).sum(0)


class PEGLinearGradNormNoop(torch.autograd.Function):
    """
    torch.autograd.Function no-op that computes the per-example gradient norms.
    In other words, it doesn't affect the forward or backward pass, just
    computes the per-example gradient norms and saves them to buffers.
    args:
        input (Tensor): the input tensor
        output (Tensor): the output tensor
    """
    @staticmethod
    def forward(ctx, input, output, weight_upegsqnorm, bias_upegsqnorm):
        # args needs to be received so we can pass gradients back, but we don't
        # need to do anything with them
        ctx.save_for_backward(input, output, weight_upegsqnorm, bias_upegsqnorm)
        return output

    @staticmethod
    def backward(ctx, grad_output):
        input, output, weight_upegsqnorm, bias_upegsqnorm = ctx.saved_tensors
        grad_input = None
        if not ctx.needs_input_grad[1]:
            grad_output = None
        # compute the per-example gradient norms and save them to the buffers
        weight_update, bias_update = pe_linear_grad_norm(input, grad_output)
        weight_upegsqnorm.add_(weight_update)
        if exists(bias_upegsqnorm):
            bias_upegsqnorm.add_(bias_update)
        return grad_input, grad_output, None, None


class PEGradNormShimLinear(ModuleWithBuffers, nn.Linear):
    def __init__(self, in_features, out_features, bias=True):
        nn.Linear.__init__(self, in_features, out_features, bias)
        self.register_upegsqnorm_buffer('weight')
        self.register_upegsqnorm_buffer('bias')

    def forward(self, input):
        output = super(PEGradNormShimLinear, self).forward(input)
        return PEGLinearGradNormNoop.apply(input, output, self.weight_upegsqnorm, self.bias_upegsqnorm)


def wgrad_plus_pe_grad_norm_highmem(g, a):
    B = a.shape[0]
    if a.ndim == 2:
        s = (g*g).sum(1) * (a*a).sum(1)
        wgrad = g.t() @ a
        bias_g = g
    elif a.ndim > 2:
        K = a.shape[-1]
        L = g.shape[-1]
        a = a.reshape(B, -1, K)
        g = g.reshape(B, -1, L)
        # materialize big tensor
        #wgrad = torch.bmm(a.transpose(1, 2), g) # B x K x L
        wgrad = torch.bmm(g.transpose(1, 2), a) # B x L x K
        # compute norms
        b = wgrad.shape[0]
        s = safe_sqnorm(wgrad.view(b, -1), dim=1)
        # collapse big tensor to gradient
        wgrad = wgrad.sum(0)
        # prepare bias gradient
        bias_g = g.sum(1)
    # compute bias pegsqnorm, it's just the incoming gradient reduced
    bias_s = safe_sum((bias_g*bias_g), 1)
    return wgrad, stack_with_ones(s, -1).sum(0), stack_with_ones(bias_s, -1).sum(0)


def at_index(x, i):
    # return element at index i or False if out of bounds
    if i > len(x) - 1:
        return False
    else:
        return x[i]


class PEGradNormLinearFunction(torch.autograd.Function):
    """
    PEGradNormLinearFunction is a torch.autograd.Function that is equivalent to
    an F.linear() operation, but with the ability to compute the GNS
    statistics at the same time.
    """
    @staticmethod
    def forward(ctx, input, weight, bias=None, weight_upegsqnorm=None, bias_upegsqnorm=None):
        ctx.save_for_backward(input, weight, bias, weight_upegsqnorm, bias_upegsqnorm)
        return F.linear(input, weight, bias)

    @staticmethod
    def backward(ctx, grad_output):
        input, weight, bias, weight_upegsqnorm, bias_upegsqnorm = ctx.saved_tensors
        # for GPU in bfloat16, grad_output will be bfloat16, so we need to
        # convert input and weight to bfloat16 as well
        input = input.to(grad_output.dtype)
        weight = weight.to(grad_output.dtype)
        grad_input = grad_weight = grad_bias = None
        if ctx.needs_input_grad[0]:
            lead_dims, j = grad_output.shape[:-1], grad_output.shape[-1]
            grad_input = grad_output.view(-1, j) @ weight
            grad_input = grad_input.view(*lead_dims, -1)
            #print(grad_input)
        if ctx.needs_input_grad[1]:
            grad_weight, weight_upegsqnorm_update, bias_upegsqnorm_update = \
                wgrad_plus_pe_grad_norm_highmem(grad_output, input)
            #print(weight_upegsqnorm, bias_upegsqnorm)
        if exists(bias) and ctx.needs_input_grad[2]:
            j = grad_output.shape[-1]
            grad_bias = grad_output.view(-1, j).sum(0)
        if bias is None:
            bias_upegsqnorm = None # no bias, no bias gradnorm
        if exists(weight_upegsqnorm):
            weight_upegsqnorm.add_(weight_upegsqnorm_update)
        if exists(bias_upegsqnorm):
            bias_upegsqnorm.add_(bias_upegsqnorm_update)
        return grad_input, grad_weight, grad_bias, None, None


class PEGradNormLinear(ModuleWithBuffers, nn.Linear):
    def __init__(self, in_features, out_features, bias=True):
        nn.Linear.__init__(self, in_features, out_features, bias)
        self.register_upegsqnorm_buffer('weight')
        if bias:
            self.register_upegsqnorm_buffer('bias')
        else:
            self.bias_upegsqnorm = None

    def forward(self, input):
        return PEGradNormLinearFunction.apply(input, self.weight, self.bias,
                                              self.weight_upegsqnorm,
                                              self.bias_upegsqnorm)


############################## Embedding ##############################


def embedding_dense_backward(g, ids, vocab_size, padding_idx, scale_grad_by_freq):
    # replicate the functionality of torch.ops.aten.embedding_dense_backward
    # with reference to: https://github.com/f-dangel/backpack/blob/1ebfb4055be72ed9e0f9d101d78806bd4119645e/backpack/core/derivatives/embedding.py#L37-L53
    # first we make a binary tensor that indicates which embedding was selected
    # on each forward pass
    b = ids.shape[0]
    delta = F.one_hot(ids, num_classes=vocab_size).float() # [*ids.shape, vocab_size]
    delta = delta.view(b, -1, vocab_size)
    # then use this to reduce
    _, d = g.shape[0], g.shape[-1]
    g = g.view(b, -1, d)
    peg = torch.einsum('bnd,bnv->vbd', g, delta)
    return peg

def embedding_pe_grad_norm(ids, g, vocab_size):
    # iterate over the batch dimension
    # shouldn't incur any additional flops in this case
    b = ids.shape[0]
    vocab = vocab_size * b
    castable = [b] + [1 for _ in ids.shape[1:]]
    # bias tensor for ids
    bias = vocab_size * torch.arange(b, device=ids.device, dtype=ids.dtype).view(*castable)
    ids = ids + bias
    # initialise output tensor
    peg = torch.ops.aten.embedding_dense_backward(g, ids, vocab, -1, False)
    #print(f"{peg.shape=}, {g.shape=}, {ids.shape=}, {vocab=}, {vocab_size=}")
    peg = peg.view(b, -1)
    #print(f"{peg=}")
    s = safe_sum(peg ** 2, 1)
    s = stack_with_ones(s, -1)
    return s.sum(0)


class PEGEmbeddingGradNormNoop(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input, output, weight, weight_upegsqnorm):
        ctx.save_for_backward(input, weight_upegsqnorm)
        ctx.vocab_size = weight.shape[0]
        return output

    @staticmethod
    def backward(ctx, grad_output):
        input, weight_upegsqnorm = ctx.saved_tensors
        grad_input = None
        if not ctx.needs_input_grad[1]:
            grad_output = None
        upegsqnorm_update = embedding_pe_grad_norm(input, grad_output, ctx.vocab_size)
        weight_upegsqnorm.add_(upegsqnorm_update)
        return grad_input, grad_output, None, None


class PEGradNormShimEmbedding(ModuleWithBuffers, nn.Embedding):
    def __init__(self, *args, **kwargs):
        nn.Embedding.__init__(self, *args, **kwargs)
        self.register_upegsqnorm_buffer('weight')

    def forward(self, input):
        output = super(PEGradNormShimEmbedding, self).forward(input)
        return PEGEmbeddingGradNormNoop.apply(input, output, self.weight, self.weight_upegsqnorm)


def embedding_peg_plus_grads(g, ids, w):
    # iterate over the batch dimension
    # shouldn't incur any additional flops in this case
    b = ids.shape[0]
    # initialise output tensor
    s = torch.zeros(b, device=ids.device)
    dw = None
    for i in range(b):
        peg = torch.ops.aten.embedding_dense_backward(g[[i]], ids[[i]],
                                                      w.shape[0], -1, False)
        if dw is None:
            dw = peg
        else:
            dw += peg
        s[i] = (peg * peg).sum()
    return dw, stack_with_ones(s, -1).sum(0)


class PEGEmbeddingGradNorm(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input, weight, pegsqnorm=None):
        ctx.save_for_backward(input, weight, pegsqnorm)
        return F.embedding(input, weight)

    @staticmethod
    def backward(ctx, grad_output):
        input, weight, pegsqnorm = ctx.saved_tensors
        grad_input = grad_weight = None
        if ctx.needs_input_grad[0]:
            raise NotImplementedError('non-differentiable in general')
        if ctx.needs_input_grad[1]:
            grad_weight, grad_buffer = embedding_peg_plus_grads(grad_output, input, weight)
            if exists(pegsqnorm):
                pegsqnorm.add_(grad_buffer)
        return grad_input, grad_weight, None


class PEGradNormEmbedding(ModuleWithBuffers, nn.Embedding):
    def __init__(self, *args, **kwargs):
        nn.Embedding.__init__(self, *args, **kwargs)
        self.register_upegsqnorm_buffer('weight')

    def forward(self, input):
        return PEGEmbeddingGradNorm.apply(input, self.weight, self.weight_upegsqnorm)


############################## LayerNorm ##############################


class ElementWiseAffine(nn.Module):
    """ The Element-wise affine part of LayerNorm """
    def __init__(self, ndim, bias):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(ndim))
        self.bias = nn.Parameter(torch.zeros(ndim)) if bias else None

    def forward(self, input):
        castable = lambda x: x.view(*[1 for _ in range(x.dim())], -1)
        out = input * castable(self.weight)
        if exists(bias):
            out += castable(self.bias)
        return out


class PEGradNormEANoop(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input, output, weight, bias, weight_upegsqnorm, bias_upegsqnorm):
        ctx.save_for_backward(input, weight_upegsqnorm, bias_upegsqnorm)
        return output

    @staticmethod
    def backward(ctx, grad_output):
        input, weight_upegsqnorm, bias_upegsqnorm = ctx.saved_tensors
        grad_input = grad_buffer = None
        if ctx.needs_input_grad[0]:
            grad_input = None # don't pass any extra gradients through
        if not ctx.needs_input_grad[1]:
            grad_output = None
        z = input
        b, d = z.shape[0], z.shape[-1]
        # reshape to 3d to make the next steps easier
        z = z.view(b, -1, d)
        g = grad_output.view(b, -1, d)
        pe_dgamma = (g * z).sum(1)
        pe_dbeta = g.sum(1)
        w_upegsqnorm = stack_with_ones((pe_dgamma * pe_dgamma).sum(1), -1).sum(0)
        b_upegsqnorm = stack_with_ones((pe_dbeta * pe_dbeta).sum(1), -1).sum(0)
        if exists(weight_upegsqnorm):
            weight_upegsqnorm.add_(w_upegsqnorm)
        if exists(bias_upegsqnorm):
            bias_upegsqnorm.add_(b_upegsqnorm)
        return grad_input, grad_output, None, None, None, None


class PEGradNormShimElementWiseAffine(ModuleWithBuffers, ElementWiseAffine):
    def __init__(self, ndim, bias):
        ElementWiseAffine.__init__(self, ndim, bias)
        self.register_upegsqnorm_buffer('weight')
        if bias:
            self.register_upegsqnorm_buffer('bias')
        else:
            self.bias_upegsqnorm = None

    def forward(self, input):
        out = super(PEGradNormShimElementWiseAffine, self).forward(input)
        return PEGradNormEANoop.apply(input, out, self.weight, self.bias,
                                      self.weight_upegsqnorm,
                                      self.bias_upegsqnorm)


class PEGradNormSeparatedLayerNorm(nn.Module):
    def __init__(self, ndim, bias):
        super().__init__()
        self.layernorm = nn.LayerNorm(ndim, elementwise_affine=False)
        self.affine = PEGradNormShimElementWiseAffine(ndim, bias)

    def forward(self, input):
        return self.affine(self.layernorm(input))



############################## Tests ##############################

class ShimLinearTest:
    def __init__(self):
        input_linear = nn.Linear(3, 3)
        self.input_state = input_linear.state_dict()

    def __call__(self, dtype, Shim, bias):
        # reset input state
        input_linear_a = nn.Linear(3, 3, bias=bias).to(dtype)
        input_linear_b = nn.Linear(3, 3, bias=bias).to(dtype)
        input_linear_a.load_state_dict(self.input_state, strict=False)
        input_linear_b.load_state_dict(self.input_state, strict=False)

        # create a linear layer
        linear = nn.Linear(3, 4, bias=bias).to(dtype)

        # create a shim linear layer
        shimlinear = Shim(3, 4, bias=bias).to(dtype)
        # set weights and biases to be the same
        with torch.no_grad():
            shimlinear.weight = nn.Parameter(linear.weight.clone())
            if bias:
                shimlinear.bias = nn.Parameter(linear.bias.clone())

        # create some input
        input = torch.randn(2, 2, 3, requires_grad=True, dtype=dtype)
        auxinput = input.clone().detach().requires_grad_(True)

        # check output
        output = linear(input_linear_a(input))
        shimoutput = shimlinear(input_linear_b(auxinput))
        assert torch.allclose(output, shimoutput)

        # check grads
        g = torch.randn_like(output)
        output.backward(g)
        shimoutput.backward(g)
        assert torch.allclose(linear.weight.grad, shimlinear.weight.grad)
        if bias:
            assert torch.allclose(linear.bias.grad, shimlinear.bias.grad)
        assert torch.allclose(input.grad, auxinput.grad)

        # check input grad
        for a, b in zip(input_linear_a.parameters(), input_linear_b.parameters()):
            assert torch.allclose(a.grad, b.grad)

        # check pegsqnorms
        if Shim in [PEGradNormShimLinear]:
            weight_pegsqnorm = shimlinear.weight_pegsqnorm()
            if bias:
                bias_pegsqnorm = shimlinear.bias_pegsqnorm()
            _weight_pegsqnorm, _bias_pegsqnorm, b = 0., 0., 0
            for i in range(input.shape[0]):
                linear.weight.grad.zero_()
                if bias:
                    linear.bias.grad.zero_()
                output = linear(input_linear_b(input[[i]]))
                output.backward(g[[i]])
                wnorm = (linear.weight.grad**2).sum()
                _weight_pegsqnorm += wnorm
                #print(f"{linear.weight.grad=} {wnorm=}")
                if bias:
                    bnorm = (linear.bias.grad**2).sum()
                    _bias_pegsqnorm += bnorm
                    #print(f"{linear.bias.grad=} {bnorm=}")
                b += 1
            _weight_pegsqnorm *= b
            _bias_pegsqnorm *= b
            assert torch.allclose(weight_pegsqnorm, _weight_pegsqnorm, rtol=1e-3), f"{shimlinear.weight_pegsqnorm()=}, {_weight_pegsqnorm=}"
            if bias:
                assert torch.allclose(bias_pegsqnorm, _bias_pegsqnorm, rtol=1e-3), f"{shimlinear.bias_pegsqnorm()=}, {_bias_pegsqnorm=}"
        print(f"{Shim} test passed {dtype=}")


class ShimEmbeddingTest:
    def __call__(self, dtype, Shim):
        # create an embedding layer
        embedding = nn.Embedding(10, 5).to(dtype)

        # create a shim embedding layer
        auxembedding = Shim(10, 5).to(dtype)
        # set weights to be the same
        with torch.no_grad():
            auxembedding.weight = nn.Parameter(embedding.weight.clone())

        # create some input
        input = torch.randint(0, 10, (2, 3))
        auxinput = input.clone().detach()

        # check output
        output = embedding(input)
        shimoutput = auxembedding(auxinput)
        assert torch.allclose(output, shimoutput)

        # check grads
        g = torch.randn_like(output)
        output.backward(g)
        shimoutput.backward(g)
        assert torch.allclose(embedding.weight.grad, auxembedding.weight.grad)

        if Shim in [PEGradNormShimEmbedding]:
            # compute pegsqnorm manually
            weight_pegsqnorm = 0.
            for i in range(2):
                embedding.weight.grad.zero_()
                output = embedding(input[[i]])
                output.backward(g[[i]])
                weight_pegsqnorm += (embedding.weight.grad**2).sum()
            weight_pegsqnorm /= 2
            weight_pegsqnorm *= 2**2
            assert torch.allclose(auxembedding.weight_pegsqnorm(), weight_pegsqnorm), f"{auxembedding.weight_pegsqnorm()=}, {weight_pegsqnorm=}"

        print(f"{Shim} test passed {dtype=}")

# reference:
class LayerNorm(nn.Module):
    """ LayerNorm but with an optional bias. PyTorch doesn't support simply bias=False """

    def __init__(self, ndim, bias):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(ndim))
        self.bias = nn.Parameter(torch.zeros(ndim)) if bias else None
        self.normalized_shape = self.weight.shape
        self.eps = 1e-5

    def forward(self, input):
        return F.layer_norm(input, self.normalized_shape, self.weight, self.bias, self.eps)


class LNShimTest:
    def __init__(self):
        input_l = nn.Linear(10, 10) # input linear layer
        self.input_state = input_l.state_dict()
    def __call__(self, dtype, LN):
        # reset input state
        input_l = nn.Linear(10, 10).to(dtype).to(dtype)
        input_l.load_state_dict(self.input_state)
        auxinput_l = nn.Linear(10, 10).to(dtype).to(dtype)
        auxinput_l.load_state_dict(self.input_state)

        # create a layernorm layer
        layernorm = LayerNorm(10, True).to(dtype)

        # create a shim layernorm layer
        auxlayernorm = LN(10, True).to(dtype)

        # create some input
        input = torch.randn(2, 5, 10, requires_grad=True).to(dtype)
        auxinput = input.clone().detach().requires_grad_(True)

        # check output
        output = layernorm(input_l(input))
        shimoutput = auxlayernorm(auxinput_l(auxinput))
        assert torch.allclose(output, shimoutput)

        # check grads
        g = torch.randn_like(output)
        output.backward(g)
        shimoutput.backward(g)
        if isinstance(auxlayernorm, PEGradNormSeparatedLayerNorm):
            aux_weight = auxlayernorm.affine.weight
            aux_bias   = auxlayernorm.affine.bias
        else:
            aux_weight = auxlayernorm.weight
            aux_bias   = auxlayernorm.bias
        err = torch.abs(layernorm.weight.grad - aux_weight.grad).max()
        if dtype == torch.float16:
            rtol, atol = 1e-2, 1e-3 # float16 is less precise
        else:
            rtol, atol = 1e-5, 1e-8
        assert torch.allclose(layernorm.weight.grad, aux_weight.grad, rtol=rtol, atol=atol), f"{layernorm.weight.grad=}, {aux_weight.grad=} {err=}"
        err = torch.abs(layernorm.bias.grad - aux_bias.grad).max()
        assert torch.allclose(layernorm.bias.grad, aux_bias.grad, rtol=rtol, atol=atol), f"{layernorm.bias.grad=}, {aux_bias.grad=} {err=}"
        for a, b in zip(input_l.parameters(), auxinput_l.parameters()):
            assert torch.allclose(a.grad, b.grad)

        if LN in [PEGradNormSeparatedLayerNorm]:
            # compute pegsqnorm manually
            weight_pegsqnorm = 0.
            bias_pegsqnorm = 0.
            for i in range(2):
                layernorm.weight.grad.zero_()
                layernorm.bias.grad.zero_()
                output = layernorm(input_l(input[[i]]))
                output.backward(g[[i]])
                weight_pegsqnorm += (layernorm.weight.grad**2).sum()
                bias_pegsqnorm += (layernorm.bias.grad**2).sum()
                #print(f"{(layernorm.weight.grad**2).sum()=}, {(layernorm.bias.grad**2).sum()=}")
            weight_pegsqnorm /= 2
            bias_pegsqnorm /= 2
            weight_pegsqnorm *= 2**2
            bias_pegsqnorm *= 2**2
            if dtype == torch.float16:
                rtol, atol = 1e-3, 1e-3
            else:
                rtol, atol = 1e-5, 1e-8
            if isinstance(auxlayernorm, PEGradNormSeparatedLayerNorm):
                aux_weight_pegsqnorm = auxlayernorm.affine.weight_pegsqnorm()
                aux_bias_pegsqnorm = auxlayernorm.affine.bias_pegsqnorm()
            else:
                aux_weight_pegsqnorm = auxlayernorm.weight_pegsqnorm()
                aux_bias_pegsqnorm = auxlayernorm.bias_pegsqnorm()
            assert torch.allclose(aux_weight_pegsqnorm, weight_pegsqnorm, rtol=rtol, atol=atol), f"{LN} {aux_weight_pegsqnorm=}, {weight_pegsqnorm=} {rtol=}, {atol=}"
            assert torch.allclose(aux_bias_pegsqnorm, bias_pegsqnorm, rtol=rtol, atol=atol), f"{aux_bias_pegsqnorm=}, {bias_pegsqnorm=}"
        print(f"{LN} test passed {dtype=}")


if __name__ == "__main__":
    # test shim linear
    import torch.nn.functional as F
    import torch
    import torch.nn as nn
    from torch.autograd import gradcheck
    # set seed
    torch.manual_seed(0)

    # test linear shims
    test = ShimLinearTest()
    for Shim in [PEGradNormShimLinear]:
        for dtype in [torch.float16, torch.float32]:
            for bias in [True, False]:
                test(dtype, Shim, bias)

    with torch.no_grad():
        for emb_dense_back in [embedding_dense_backward]:
            # test manual embedding backward
            g = torch.randn(2, 3, 5)
            ids = torch.randint(0, 10, (2, 3))
            weight = torch.randn(10, 5)
            peg = emb_dense_back(g, ids, 10, -1, False)
            dw = peg.sum(1) # sum over batch dimension
            _dw = torch.ops.aten.embedding_dense_backward(g, ids, 10, -1, False)
            assert torch.allclose(dw, _dw)


    # test embedding shims
    for Shim in [PEGradNormShimEmbedding]:
        for dtype in [torch.float16, torch.float32]:
            ShimEmbeddingTest()(dtype, Shim)


    # test layernorm shims
    test = LNShimTest()
    for LN in [PEGradNormSeparatedLayerNorm]:
        for dtype in [torch.float16, torch.float32]:
            test(dtype, LN)

    # create a random input and weight
    input = torch.randn(3, 5, requires_grad=True)
    weight = torch.randn(4, 5, requires_grad=True)
    bias = torch.randn(4, requires_grad=True)

    # test
    def test_func(f):
        # zero gradients if they exist
        for t in [input, weight, bias]:
            if t.grad is not None:
                t.grad.zero_()
        # compute the output
        output = f(input, weight, bias)
        # compute the gradient of the output
        output.sum().backward()
        return output, input.grad, weight.grad, bias.grad
    a = test_func(PEGradNormLinearFunction.apply)
    b = test_func(F.linear)
    for i in range(len(a)):
        assert torch.allclose(a[i], b[i]), i

    # create a random input and weight
    input = torch.randn(3, 2, 5, requires_grad=True)
    weight = torch.randn(4, 5, requires_grad=True)
    bias = torch.randn(4, requires_grad=True)
    # test
    def test_func(f):
        # zero gradients if they exist
        for t in [input, weight, bias]:
            if t.grad is not None:
                t.grad.zero_()
        # compute the output
        output = f(input, weight, bias)
        # compute the gradient of the output
        output.sum().backward()
        return output, input.grad, weight.grad, bias.grad
    a = test_func(PEGradNormLinearFunction.apply)
    b = test_func(F.linear)
    for i in range(len(a)):
        assert torch.allclose(a[i], b[i]), i

    for bias in [False, True]:
        gns_linear = PEGradNormLinear(5, 4, bias=bias)

        out = gns_linear(input)
        out.sum().backward()

        w_pegsqgradnorm = torch.empty(3)
        b_pegsqgradnorm = torch.empty(3)
        for b in range(3):
            # zero grads
            for t in [gns_linear.weight] + ([gns_linear.bias] if bias else []):
                if t.grad is not None:
                    t.grad.zero_()
            # compute the output
            output = F.linear(input[[b]], gns_linear.weight, gns_linear.bias)
            # compute the gradient of the output
            output.sum().backward()
            # compute the pe gradnorm
            w_pegsqgradnorm[b] = torch.einsum('ij,ij->', gns_linear.weight.grad, gns_linear.weight.grad)
            if bias:
                b_pegsqgradnorm[b] = torch.einsum('i,i->', gns_linear.bias.grad, gns_linear.bias.grad)
        w_pegsqgradnorm = w_pegsqgradnorm.mean(dim=0, keepdim=True) * 9
        # print(w_pegsqgradnorm)
        assert torch.allclose(gns_linear.weight_pegsqnorm(), w_pegsqgradnorm), (gns_linear.weight_pegsqnorm(), w_pegsqgradnorm)
        b_pegsqgradnorm = b_pegsqgradnorm.mean(dim=0, keepdim=True) * 9
        if bias:
            assert torch.allclose(gns_linear.bias_pegsqnorm(), b_pegsqgradnorm)
        print(f"PEGradNormLinear test with bias={bias} passed")

   def accumulate_sqgradnorm_buffers(model):
        total = 0
        for m in model.modules():
            if hasattr(m, 'named_buffers_with_marker'):
                for name, buffer in m.named_buffers_with_marker('is_pegsqnorm').items():
                    #print(f"{name=} {buffer=} {torch.prod(buffer)}")
                    total += torch.prod(buffer)
        return total

    # integration test, in a network
    class Net(nn.Module):
        def __init__(self):
            super(Net, self).__init__()
            self.fc1 = PEGradNormLinear(5, 4)
            self.fc2 = PEGradNormLinear(4, 3)

        def forward(self, x):
            x = self.fc1(x)
            x = self.fc2(x)
            return x

    net = Net()
    #print(list(net.named_parameters()))
    #net.to("mps")
    #assert False
    w_pegsqgradnorm = torch.empty(3)
    for b in range(3):
        # zero grads
        for p in net.parameters():
            if p.grad is not None:
                p.grad.zero_()
        # compute the output
        output = net(input[[b]])
        # compute the gradient of the output
        output.sum().backward()
        # compute the pe gradnorm
        s = net.fc1.weight.grad.pow(2).sum() + net.fc2.weight.grad.pow(2).sum()
        s += net.fc1.bias.grad.pow(2).sum() + net.fc2.bias.grad.pow(2).sum()
        w_pegsqgradnorm[b] = s
        #print(f"in loop {s=}")
    # print(f"{w_pegsqgradnorm=}")
    #print(f"{w_pegsqgradnorm=}")
    #print(f"{w_pegsqgradnorm=}")
    w_pegsqgradnorm = w_pegsqgradnorm.mean(dim=0, keepdim=True) * 9
    zero_sqgradnorm_buffers(net)
    #print(f"{accumulate_sqgradnorm_buffers(net)=}")
    output = net(input)
    output.sum().backward()
    #print(f"{accumulate_sqgradnorm_buffers(net)=}")
    assert torch.allclose(accumulate_sqgradnorm_buffers(net), w_pegsqgradnorm),\
           (accumulate_sqgradnorm_buffers(net), w_pegsqgradnorm)

    # test embedding
    emb = PEGradNormEmbedding(5, 4)
    w_pegsqgradnorm = torch.empty(3)
    input = torch.randint(5, (3, 2))
    for b in range(3):
        # zero grads
        for p in emb.parameters():
            if p.grad is not None:
                p.grad.zero_()
        # compute the output
        output = emb(input[[b]])
        # compute the gradient of the output
        output.sum().backward()
        # compute the pe gradnorm
        w_pegsqgradnorm[b] = emb.weight.grad.pow(2).sum()
    w_pegsqgradnorm = w_pegsqgradnorm.mean(dim=0, keepdim=True)

    # test gradient accumulation
    net = PEGradNormLinear(5, 4)
    loss = lambda x: torch.square(x).mean()
    data = [torch.randn(3, 5) for _ in range(3)]
    total = 0.
    for d in data:
        output = net(d)
        l = loss(output) / len(data)
        total += l.item()
        #print(f"{total=}")
        l.backward()
    weight_pegsqnorm = net.weight_pegsqnorm()
    #print(f"{weight_pegsqnorm=}")
    bias_pegsqnorm = net.bias_pegsqnorm()
    zero_sqgradnorm_buffers(net)
    #print(net.weight_upegsqnorm.grad)
    output = net(torch.cat(data, 0))
    l = loss(output)
    #print(l)
    l.backward()
    #print(net.weight_upegsqnorm.grad)
    assert torch.allclose(net.weight_pegsqnorm(), weight_pegsqnorm), f"{net.weight_pegsqnorm()=} {weight_pegsqnorm=}"
    print("Success!")
