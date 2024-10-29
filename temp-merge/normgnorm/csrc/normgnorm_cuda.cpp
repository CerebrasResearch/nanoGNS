#include <torch/extension.h>

extern std::vector<torch::Tensor> layernorm_fwd(
    torch::Tensor x,
    torch::Tensor weight,
    torch::Tensor bias,
    float eps
);

extern std::vector<torch::Tensor> layernorm_bwd(
    torch::Tensor grad_input,
    torch::Tensor x,
    torch::Tensor weight,
    torch::Tensor mean,
    torch::Tensor rstd
);


PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.doc() = "Norm functions with backwards giving per example grad norms"; // optional module docstring
    
    m.def("layernorm_fwd", layernorm_fwd);
    m.def("layernorm_bwd", layernorm_bwd);
}
 
