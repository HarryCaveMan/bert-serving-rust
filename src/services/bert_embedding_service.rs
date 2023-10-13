use actix_web::{post,web,HttpResponse};
use rust_bert::{RustBertError};
use rust_bert::pipelines::sentence_embeddings::{Embedding};
use serde::{Serialize,Deserialize};
use log::{debug,error};
use std::time::{Instant};

use super::bert::bert_embedding_model::BertEmbeddingModel;

#[derive(Serialize,Deserialize)]
struct EmbeddingRequest {
    id: u32,
    sentences: Vec<String>
}
#[derive(Serialize,Deserialize)]
struct EmbeddingResponse {
    id: u32,
    embeddings: Vec<Embedding>
}
#[derive(Serialize,Deserialize)]
struct ErrorResponse {
    id: u32,
    message: String
}

#[post("/get_embeddings")]
async fn get_embeddings(model: web::Data<BertEmbeddingModel>, req: web::Json<EmbeddingRequest>) -> HttpResponse {
    let id: u32 = req.id;
    debug!("Starting inference...");
    let start: Instant = Instant::now();
    let model_result: Result<Vec<Embedding>, RustBertError> = model.get_embeddings(&req.sentences);
    debug!("Done! Took {:?}ms",start.elapsed().as_millis());
    match model_result {
        Ok(embeddings) => {
            debug!("Embeddings shape: [{:?}{:?}]",embeddings.len(),embeddings[0].len());
            HttpResponse::Ok().json(EmbeddingResponse{id:id,embeddings:embeddings})
        },
        Err(err) => {
            error!("{:?}",err);
            HttpResponse::InternalServerError().json(
                ErrorResponse{
                    id:id,
                    message:String::from("Internal call to inference model failed!")
                }
            )
        }
    }
}