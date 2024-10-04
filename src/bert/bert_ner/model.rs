use rust_bert::pipelines::ner::{NERModel,Entity};
use rust_bert::{RustBertError};
use log::{info,debug};
use tch::{Device};
use super::builder::{NERBuilder};

pub struct BertNERModel {
    model: NERModel,
    spans: bool
}

impl BertNERModel {
    pub fn new_from_file(file: &str, spans: Option<bool>) -> Result<Self,RustBertError> {
        let device = Device::cuda_if_available();
        info!("Found device: {:?}",device);
        let model_result: Result<NERModel, RustBertError> = NERBuilder::local(file)
            .with_device(device)
            .create_model();
        match model_result {
            Ok(model) => Ok(
                Self{
                    model: model,
                    spans: match spans{Some(flag)=>flag,None=>false}
                }
            ),
            Err(err) => Err(err)
        }
    }
    pub fn predict(&self,sentences: &Vec<String>) -> Vec<Vec<Entity>> {
        if self.bio_enabled {
            self.model.predict_full_entities(sentences)
            // debug!("{:?}",tagged_tokens);
            // tag_spans_bio(sentences,&tagged_tokens)
        }
        else {
            self.model.predict(sentences)
        }        
    }
}