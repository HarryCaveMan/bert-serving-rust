FROM nvidia/cuda:11.8.0-base-ubuntu22.04

RUN apt -y update && apt -y upgrade &&\
    apt -y install \
      git \
      git-lfs \
      python3 \
      python3-pip \
    &&\
    pip3 install \
      torch==2.0 \
      jupyterlab \
      scikit-learn \
      matplotlib \
    &&\
    curl -sSf https://sh.rustup.rs| bash -s -- y

WORKDIR /opt

CMD ["jupyter","lab","--allow-root","--ip","0.0.0.0"]