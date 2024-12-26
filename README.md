# Pytorch CPP Samples
A collection of some introductory pytorch cpp toy projects

Heavily inspired by:
- https://youtube.com/playlist?list=PLZAGo22la5t4UWx37MQDpXPFX3rTOGO3k&si=EQW7jhzEaQEQyVGK
- https://pytorch.org/cppdocs/installing.html

## Setup

Download the Pytorch C++ LibTorch compatible with your CUDA version from here:

https://pytorch.org/get-started/locally/

Change the `CUDA version` in the `Dockerfile` and build the docker image:
```bash
docker build -f Dockerfile -t torchcpp .
```

## Usage

Enter the docker image

```bash
docker run --rm -it --gpus all -v $(pwd):/workspace torchcpp bash
```

Inside the root directory create a build dir and build the project:

```bash
mkdir build && cd build
cmake -DCMAKE_PREFIX_PATH=/absolute/path/to/libtorch ..
make
./your_executable
```


## TODO


## Contribution
Hamid Mohammadi: <hamid4@ualberta.ca>
