use super::{SentenceEmbeddingsBuilder};
use tch::{no_grad};

struct RankedAnswer {
    text: String,
    score: f32
}

pub struct RankedResults {
    questions: Vec<String>,
    ranked_answers: Vec<RankedAnswer>
}

pub struct BertRerankingModel {
    model: BertSequenceClassificationModel
}



impl BertRerankingModel {
    pub fn new_from_file(file: &str) -> Result<Self,RustBertError> {
        let device = Device::cuda_if_available();
        info!("Found device: {:?}",device);
        let model_result: Result<SequenceClassificationModelT>, RustBertError> = SequenceClassificationBuilder::local(file)
            .with_device(device)
            .create_model();
        match model_result {
            Ok(model) => Ok(Self{model: model}),
            Err(err) => Err(err)
        }
    }
}