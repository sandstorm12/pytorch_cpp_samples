import torch

from classifier import ClassifierMNIST


if __name__ == "__main__":
    model = ClassifierMNIST()
    model.load_state_dict(torch.load("model.pth", weights_only=True))

    example_input = torch.rand(32, 1, 28, 28)
    
    script_module = torch.jit.trace(model, example_input)
    script_module.save("model.pt")
