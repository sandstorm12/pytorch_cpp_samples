#include <iostream>
#include <cuda_runtime.h>
#include <torch/torch.h>

int main(){
    torch::Device device(torch::kCPU);
    if (torch::cuda::is_available()) {
        device = torch::Device(torch::kCUDA, 0);
    }

    torch::Tensor x = torch::randn({3, 3}, device);
    std::cout << x << std::endl;

    return 0;
} 