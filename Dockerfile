ARG SERVICE=bert_embedding_service

FROM nvidia/cuda:11.8.0-base-ubuntu22.04 as base

WORKDIR /opt

RUN apt -y update && apt -y upgrade &&\
    apt -y install \
        pkg-config libssl-dev tar unzip

ENV LIBTORCH=/opt/libtorch
ENV LD_LIBRARY_PATH="${LIBTORCH}/lib:$LD_LIBRARY_PATH"

FROM base as builder

RUN apt -y install curl && \
    curl -sSf https://sh.rustup.rs | sh -s -- -y && \
    curl -o libtorch.zip https://download.pytorch.org/libtorch/cu118/libtorch-cxx11-abi-shared-with-deps-2.0.0%2Bcu118.zip &&\
    unzip libtorch.zip
  
ENV PATH="/root/.cargo/bin:$PATH"
COPY . .
RUN cargo build $SERVICE --release

FROM base

COPY --from=builder /opt/libtorch /opt/libtorch
COPY --from=builder /opt/target/release /opt/server

RUN mv /opt/server/$SERVICE /opt/server/serve

ENV PATH="/opt/server:$PATH"

cmd ["serve"]