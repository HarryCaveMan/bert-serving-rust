FROM nvidia/cuda:11.8.0-base-ubuntu22.04 as base
ARG SERVICE=bert_embedding_service
ENV SERVICE=$SERVICE
ENV LIBTORCH=/opt/libtorch
ENV LD_LIBRARY_PATH="${LIBTORCH}/lib:$LD_LIBRARY_PATH"

WORKDIR /opt
# ALL Runtime apt dependencies are installed here
RUN apt -y update && apt -y upgrade &&\
    apt -y install \
        pkg-config libssl-dev libgomp1 acl tar unzip gzip

FROM base as libtorch-base
# Install libtorch binary into its own layer
RUN apt -y install curl &&\
    curl --proto '=https' --tlsv1.2 -o libtorch.zip https://download.pytorch.org/libtorch/cu118/libtorch-cxx11-abi-shared-with-deps-2.2.0%2Bcu118.zip &&\
    unzip libtorch.zip

FROM libtorch-base as builder
# Stack application build on top of libtorch layer
RUN apt -y install build-essential && curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y  
  
ENV PATH="/root/.cargo/bin:$PATH"
COPY . .
RUN cargo build --release --bin $SERVICE

FROM base
# Start with fresh base layer and bopy over build artifacts
COPY --from=builder /opt/libtorch /opt/libtorch
COPY --from=builder /opt/target/release /opt/server
# Setup app-user and add shim for sagemaker compatability makes the entrypoint command "serve"
RUN mv /opt/server/$SERVICE /opt/server/serve &&\
    useradd app-user -u 1000 -M -s /bin/false &&\
    setfacl -m user:app-user:r-x /opt/server/serve
ENV PATH="/opt/server:$PATH"

USER app-user

CMD ["serve"]