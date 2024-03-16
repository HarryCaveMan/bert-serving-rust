use rust_bert::{RustBertError};
use log::{info};
use tch::{Device};
use super::bert_reranking_model::{RerankingModel};
use super::builder::{RerankingBuilder};

pub use super::bert_reranking_model::{RankedResults};

pub struct BertRerankingModel {
    model: RerankingModel
}

impl BertRerankingModel {
    pub fn new_from_file(file: &str) -> Result<Self,RustBertError> {
        let device = Device::cuda_if_available();
        info!("Found device: {:?}",device);
        let model_result: Result<RerankingModel, RustBertError> = RerankingBuilder::local(file)
            .with_device(device)
            .create_model();
        match model_result {
            Ok(model) => Ok(Self{model: model}),
            Err(err) => Err(err)
        }
    }
    pub fn predict(&self, queries: &Vec<String>, results_set: &Vec<Vec<String>>, logit_index_threshold: i64) -> RankedResults {
        let query_refs: Vec<&str> = queries.iter().map(|s| s.as_ref()).collect();
        let results_refs: Vec<Vec<&str>> = results_set.iter().map(|v| v.iter().map(AsRef::as_ref).collect())
        .collect();
        self.model.predict(&query_refs,&results_refs,logit_index_threshold)
    }
}