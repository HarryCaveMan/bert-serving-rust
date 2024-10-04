use rust_bert::pipelines::ner::{NERModel,Entity};
use rust_bert::{RustBertError};
use log::{info,debug};
use tch::{Device};
use super::builder::{NERBuilder};
use super::spans::{tag_spans_bio};

pub struct BertNERModel {
    model: NERModel,
    bio_enabled: bool
}

impl BertNERModel {
    pub fn new_from_file(file: &str, bio_enabled: Option<bool>) -> Result<Self,RustBertError> {
        let device = Device::cuda_if_available();
        info!("Found device: {:?}",device);
        let model_result: Result<NERModel, RustBertError> = NERBuilder::local(file)
            .with_device(device)
            .create_model();
        match model_result {
            Ok(model) => Ok(
                Self{
                    model: model,
                    bio_enabled: match bio_enabled{Some(flag)=>flag,None=>false}
                }
            ),
            Err(err) => Err(err)
        }
    }
    pub fn predict(&self,sentences: &Vec<String>) -> Vec<Vec<Entity>> {
        if self.bio_enabled {
            let tagged_tokens = self.model.predict(sentences);
            debug!("{:?}",tagged_tokens);
            tag_spans_bio(sentences,&tagged_tokens)
        }
        else {
            self.model.predict(sentences)
        }        
    }
}