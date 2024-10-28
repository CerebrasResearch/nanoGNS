import torch
from torch.nn import functional as F
from torch.utils.cpp_extension import load

normgnorm = load(name='normgnorm', sources=['csrc/normgnorm.cu', 'csrc/normgnorm.cpp'])

z = 4096
sl = 512
x = torch.randn(16, z).cuda()
w = torch.rand(z).cuda()
b = torch.rand(z).cuda()

y = F.layer_norm(x, [z], weight=w, bias=b, eps=1e-5)

y2, m, r = normgnorm.layernorm_fwd(x, w, b, 1e-5)
print((y - y2).abs().max())
assert torch.allclose(y, y2, atol=1e-6)


x = torch.randn(2, sl, z).cuda()
w = torch.rand(z).cuda()
b = torch.rand(z).cuda()

w.requires_grad_(True)
b.requires_grad_(True)

pegs = []
for i in range(2):
    for j in range(sl):
        xi = x[i, j].clone().detach()
        yi = F.layer_norm(xi, [z], weight=w, bias=b, eps=1e-5)
        yi.sum().backward()
        pegs.append(w.grad.clone())
        w.grad.zero_()


x.requires_grad_(True)
y3 = F.layer_norm(x, [z], weight=w, bias=b, eps=1e-5)
y3.sum().backward()


pegs = torch.stack(pegs).reshape(2, sl, z)

_, m, r = normgnorm.layernorm_fwd(x.reshape(-1, z), w, b, 1e-5)
m = m.reshape(x.shape[:-1])
r = r.reshape(x.shape[:-1])
go, wgo, bgo = normgnorm.layernorm_bwd(torch.ones(2, sl, z).cuda(), x, w, m, r)
print((wgo - pegs.sum(1)).abs().max())
print((x.grad - go).abs().max())
assert torch.allclose(wgo, pegs.sum(1), atol=1e-5)
assert torch.allclose(x.grad, go, atol=1e-6)

