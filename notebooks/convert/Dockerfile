FROM nvidia/cuda:11.8.0-cudnn8-devel-ubuntu22.04

ENV LIBTORCH=/opt/libtorch
ENV LD_LIBRARY_PATH="${LIBTORCH}/lib:$LD_LIBRARY_PATH"
ENV PATH="/root/.cargo/bin:$PATH"

WORKDIR /opt
RUN apt -y update && apt -y upgrade &&\
    apt -y install \
      libssl-dev \
      pkg-config \
      git \
      python3 \
      python3-pip \
      unzip \
      curl \
    &&\
    curl --proto '=https' --tlsv1.2 -o libtorch.zip https://download.pytorch.org/libtorch/cu118/libtorch-cxx11-abi-shared-with-deps-2.4.0%2Bcu118.zip &&\
    unzip libtorch.zip && rm -f libtorch.zip &&\
    pip3 install \
      torch>=2.4 \
      transformers \
      diffusers \
      accelerate \
      pillow \
      scikit-learn \
    &&\
    curl --proto '=https' --tlsv1.2 https://sh.rustup.rs| bash -s -- -y &&\
    rustup component add rust-src

RUN git clone https://github.com/guillaume-be/rust-bert.git 

WORKDIR /opt/workspace

ENTRYPOINT ["python3","/opt/rust-bert/utils/convert_model.py"]