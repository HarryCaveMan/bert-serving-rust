use rust_bert::pipelines::sentence_embeddings::{Embedding,SentenceEmbeddingsModel,SentenceEmbeddingsBuilder};
use rust_bert::{RustBertError};
use tch::{Device};

pub struct BertEmbeddingModel {
    model: SentenceEmbeddingsModel
}

impl BertEmbeddingModel {
    pub fn new_from_file(file: &str) -> Result<Self,RustBertError> {
        let model_result: Result<SentenceEmbeddingsModel, RustBertError> = SentenceEmbeddingsBuilder::local(file)
            .with_device(Device::cuda_if_available())
            .create_model();
        match model_result {
            Ok(model) => Ok(Self{model: model}),
            Err(err) => Err(err)
        }
    }
    pub fn get_embeddings(&self,sentences: &Vec<String>) -> Result<Vec<Embedding>, RustBertError> {
        self.model.encode(sentences)
    }
}