#include <iostream>
#include <cuda_runtime.h>
#include <torch/torch.h>

int main(){
    // Check if CUDA is available
    torch::Device device(torch::kCPU);
    if (torch::cuda::is_available()) {
        // Select the first GPU
        device = torch::Device(torch::kCUDA, 0);
    }

    // Create a random tensor
    torch::Tensor x = torch::randn({3, 3}, device);
    std::cout << x << std::endl;

    return 0;
} 