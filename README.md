# Bert Serving Rust
A Crate for serving Transformer models written by a noob rustacean for practice, meant to actually be useful and robust.

## Background
Heavily dependent on [`rust_bert`](https://github.com/guillaume-be/rust-bert/tree/main), which itself is inspired by HuggingFace Transformers. This crate is essentially a platform for building the `rust_bert` pipelines into `actix_web` microservices.

The ultimate near-term goal of this project is to support all the [pipelines](https://github.com/guillaume-be/rust-bert/blob/main/README.md#ready-to-use-pipelines) currently supported by `rust_bert`. The long-term would not be limited to these, because botht this crate and `rust_bert` offer flexibility and extensibility.

## In Progress Features (17 October 2023)
- Working on reranking
- Working on performance/load benchmarks
- Sequence Classification added 17 October 2023
- NER support added 15 October 2023
- (Paused 14 October 2023) Workig on support for remote models from HF Hub

## Current Services (17 October 2023)
- [bert_embedding_service](doc/bert_embedding_service.md)
- [bert_ner_service](doc/bert_ner_service.md)
- [bert_sequence_classification_service](doc/bert_sequence_classification_service.md)

## Building a service image
All service images use the same `Dockerfile`. You can select which service to build using the `SERVICE` build arg IE:
```sh
docker build --build-arg SERVICE=bert_embedding_service .
```

## Launching a service image
The models are not included in the image and must currently be provided as a volume mount. Support for remote models on HF hub will be added very soon. S3 will be next. Followed by other cloud buckets/blobstores.

Once you have downloaded byour HF model package, you can share it with the container via a volume mount and setting the `MODEL_PATH` environment variable to the mounted path.

There is a convenience sctipt [`launch-in-docker`](launch-in-docker) which allows you to pass a `local_model_path` and a `container_model_path` and will automatically take care of both sharing the volume and setting `MODEL_PATH` correctly. The [`launch-in-docker`](launch-in-docker) script takes two additional optional parameters for your convenience:
- `rebuild`: Whether or not to rebuild `image_tag` the image even if it exists
- `image_tag`: The tag that will be given to the local image, if it doesn't exist, or if `rebuild=yes`, it wil be built
### Exapmle
```sh
# You would replace these with paths that matched your model
local_path="$(pwd)/notebooks/models/all-MiniLM-L12-v2"
container_path="/opt/ml/models/all-MiniLM-L12-v2"
./launch-in-docker \
    service=bert_embedding_service \
    image=embedding_test \
    local_model_path=local_path \
    container_model_path=container_path
```
This would serve `all-MiniLM-L12-v2` embeddings at `localhost:5000/embeddings/encode`