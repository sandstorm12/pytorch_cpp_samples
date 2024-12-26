#include <torch/torch.h>

struct CustomModule : torch::nn::Module {
    torch::Tensor W;
    torch::Tensor b;
    
    CustomModule(int64_t N, int64_t M) {
        // Register a weight and bias for the linear layer
        W = register_parameter("W", torch::randn({N, M}));
        b = register_parameter("b", torch::randn({M}));
    }

    torch::Tensor forward(torch::Tensor x) {
        return torch::addmm(b, x, W);
    }
};

int main() {
    torch::Device device(torch::kCPU);
    // Check if CUDA is available
    if (torch::cuda::is_available()) {
        // Get the number of available GPUs
        int num_gpus = torch::cuda::device_count();
        std::cout << "Number of GPUs: " << num_gpus << std::endl;
        // Use the last available GPU (just for the sake of it)
        device = torch::Device(torch::kCUDA, num_gpus - 1);
    }

    // Create a linear layer with 32 input dim and 64 output dim
    CustomModule layer(32, 64);
    // Move the layer to the selected device
    layer.to(device);

    // Printing all the parameters in the layer
    for (const auto& p : layer.parameters()) {
        std::cout << p << std::endl;
    }

    return 0;
}