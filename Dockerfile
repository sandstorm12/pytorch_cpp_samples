FROM nvidia/cuda:12.1.0-devel-ubuntu22.04

RUN apt -y update && \
    apt -y install cmake g++