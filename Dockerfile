

FROM nvidia/cuda:11.8.0-base-ubuntu22.04 as base
ARG SERVICE=bert_embedding_service
ENV SERVICE=$SERVICE
ENV LIBTORCH=/opt/libtorch
ENV LD_LIBRARY_PATH="${LIBTORCH}/lib:$LD_LIBRARY_PATH"

WORKDIR /opt

RUN apt -y update && apt -y upgrade &&\
    apt -y install \
        pkg-config libssl-dev libgomp1 tar unzip

FROM base as libtorch-base

RUN apt -y install curl &&\
    curl -o libtorch.zip https://download.pytorch.org/libtorch/cu118/libtorch-cxx11-abi-shared-with-deps-2.0.0%2Bcu118.zip &&\
    unzip libtorch.zip

FROM libtorch-base as builder

RUN apt -y install build-essential && curl -sSf https://sh.rustup.rs | sh -s -- -y   
  
ENV PATH="/root/.cargo/bin:$PATH"
COPY . .
RUN cargo build --release --bin $SERVICE

FROM base

COPY --from=builder /opt/libtorch /opt/libtorch
COPY --from=builder /opt/target/release /opt/server

RUN mv /opt/server/$SERVICE /opt/server/serve

ENV PATH="/opt/server:$PATH"

cmd ["serve"]