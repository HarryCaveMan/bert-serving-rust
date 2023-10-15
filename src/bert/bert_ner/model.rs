use rust_bert::pipelines::ner::{NERModel,Entity};
use rust_bert::{RustBertError};
use log::{info};
use tch::{Device};
use super::builder::{NERBuilder};

pub struct BertNERModel {
    model: NERModel
}

impl BertNERModel {
    pub fn new_from_file(file: &str) -> Result<Self,RustBertError> {
        let device = Device::cuda_if_available();
        info!("Device is CUDA: {:?}",device.is_cuda());
        let model_result: Result<NERModel, RustBertError> = NERBuilder::local(file)
            .with_device(device)
            .create_model();
        match model_result {
            Ok(model) => Ok(Self{model: model}),
            Err(err) => Err(err)
        }
    }
    pub fn predict(&self,sentences: &Vec<String>) -> Vec<Vec<Entity>> {
        self.model.predict(sentences)
    }
}