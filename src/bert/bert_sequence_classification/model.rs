use rust_bert::pipelines::sequence_classification::{SequenceClassificationModel,Label};
use rust_bert::{RustBertError};
use log::{info};
use tch::{Device};
use super::builder::{SequenceClassificationBuilder};

trait SequenceClassifierReranker {
    pub fn rerank_one(&self,query: &str, results: &Vec<&str>) {}
}

pub struct BertSequenceClassificationModel {
    model: SequenceClassificationModel
}



impl BertSequenceClassificationModel {
    pub fn new_from_file(file: &str) -> Result<Self,RustBertError> {
        let device = Device::cuda_if_available();
        info!("Found device: {:?}",device);
        let model_result: Result<SequenceClassificationModel, RustBertError> = SequenceClassificationBuilder::local(file)
            .with_device(device)
            .create_model();
        match model_result {
            Ok(model) => Ok(Self{model: model}),
            Err(err) => Err(err)
        }
    }
    pub fn predict(&self,sentences: &Vec<String>) -> Vec<Label> {
        let sentence_refs: Vec<&str> = sentences.iter().map(|s| s.as_ref()).collect();
        self.model.predict(&sentence_refs)
    }
}