use actix_web::{post,web,HttpResponse};
use rust_bert::pipelines::sequence_classification::{Label};
use serde::{Serialize,Deserialize};
use log::{debug};
use std::time::{Instant};
use super::bert::bert_sequence_classification::model::{BertSequenceClassificationModel};

#[derive(Serialize,Deserialize)]
struct SequenceClassificationRequest {
    crid: u32,
    sentences: Vec<String>
}
#[derive(Serialize,Deserialize)]
struct SequenceClassificatioResponse {
    crid: u32,
    labels: Vec<Label>
}

#[post("/predict")]
async fn predict(model: web::Data<BertSequenceClassificationModel>, req: web::Json<SequenceClassificationRequest>) -> HttpResponse {
    let crid: u32 = req.crid;
    debug!("Starting inference...");
    let start: Instant = Instant::now();
    let labels: Vec<Label> = model.predict(&req.sentences);
    debug!("Done! Took {:?}ms",start.elapsed().as_millis());    
    debug!("Labels:\n{:?}",labels);
    HttpResponse::Ok().json(SequenceClassificatioResponse{crid:crid,labels:labels})
}