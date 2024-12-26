# Pytorch CPP Samples
A collection of some introductory pytorch cpp toy projects

Heavily inspired by:
- https://youtube.com/playlist?list=PLZAGo22la5t4UWx37MQDpXPFX3rTOGO3k&si=EQW7jhzEaQEQyVGK
- https://pytorch.org/cppdocs/installing.html

## Setup

Change the `CUDA version` in the `Dockerfile` and build the docker image:
```bash
docker build -f Dockerfile -t torchcpp .
```

## Usage

Enter the docker image

```bash
docker run --rm -it --gpus all -v $(pwd):/workspace torchcpp bash
```


## TODO


## Contribution
Hamid Mohammadi: <hamid4@ualberta.ca>
