use actix_web::{post,web,HttpResponse};
use rust_bert::pipelines::ner::{Entity};
use serde::{Serialize,Deserialize};
use log::{debug};
use std::time::{Instant};

use super::bert::bert_ner::model::{BertNERModel};

#[derive(Serialize,Deserialize)]
struct NERRequest {
    crid: u32,
    sentences: Vec<String>
}
#[derive(Serialize,Deserialize)]
struct NERResponse {
    crid: u32,
    entities: Vec<Vec<Entity>>
}

#[post("/predict")]
async fn predict(model: web::Data<BertNERModel>, req: web::Json<NERRequest>) -> HttpResponse {
    let crid: u32 = req.crid;
    debug!("Starting inference...");
    let start: Instant = Instant::now();
    let entities: Vec<Vec<Entity>> = model.predict(&req.sentences);
    debug!("Done! Took {:?}ms",start.elapsed().as_millis());    
    debug!("Entities:\n{:?}",entities);
    HttpResponse::Ok().json(NERResponse{crid:crid,entities:entities})
}