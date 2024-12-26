#include <torch/torch.h>
#include <torch/script.h>

torch::Device get_device() {
    // Check if CUDA is available
    torch::Device device(torch::kCPU);
    if (torch::cuda::is_available()) {
        // Select the first GPU
        device = torch::Device(torch::kCUDA, 0);
    }

    return device;
}

int main() {
    torch::Device device = get_device();

    // Load the script model and move to device
    torch::jit::script::Module module;
    module = torch::jit::load("../model.pt");
    module.to(device);
    module.eval();

    // Create a random input tensor and move to device
    torch::Tensor input = torch::randn({32, 1, 28, 28});
    input = input.to(device);
    std::cout << "Input:" << input.sizes() << " " << input.dtype()
        << " " << input.device() << std::endl;
    
    // Execute the model and move the output to CPU
    torch::Tensor output = module.forward({input}).toTensor().cpu();
    std::cout << "Output:" << output.sizes() << " " << output.dtype()
        << " " << output.device() << std::endl;

    return 0;
}