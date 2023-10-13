use actix_web::{get,web,App,HttpServer,HttpResponse};
use actix_web::middleware::{Logger};
use log::{info};
use std::{env};

use bert_serving_rust::bert::bert_embedding_model::{BertEmbeddingModel};
use bert_serving_rust::services::bert_embedding_service::{get_embeddings};

#[get("/ping")]
async fn ping() -> HttpResponse {
    HttpResponse::Ok().body("Ready!!")
}

#[actix_web::main]
async fn main() -> std::io::Result<()> {
    env_logger::init();
    HttpServer::new( move || {
        let model_path: &str = &match env::var("MODEL_PATH") {
            Ok(val) => val,
            Err(err) => String::from("notebooks/models/all-MiniLM-L12-v2")
        };
        let model: BertEmbeddingModel = BertEmbeddingModel::new_from_file(model_path).unwrap();
        info!("Starting service on port 5000...");
        App::new()
            .service(
                web::scope("/embeddings")
                    .app_data(web::Data::new(model))
                    .service(get_embeddings)
                    .service(ping)
            )
            .wrap(Logger::default())
    })
    .bind(("0.0.0.0",5000))?
    .run()
    .await
}