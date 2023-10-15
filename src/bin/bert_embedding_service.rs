use actix_web::{get,web,App,HttpServer,HttpResponse};
use actix_web::middleware::{Logger};
use log::{info};
use std::{env};

use bert_serving_rust::bert::bert_embedding_model::{BertEmbeddingModel};
use bert_serving_rust::services::bert_embedding_service::{encode};

#[get("/ping")]
async fn ping() -> HttpResponse {
    HttpResponse::Ok().body("Ready!!")
}

#[actix_web::main]
async fn main() -> std::io::Result<()> {
    env_logger::init();
    let num_web_workers: usize = match env::var("NUM_WEB_WORKERS") {
        Ok(val) => val.parse<usize>().unwrap(),
        Err(err) => {
            error!("{:?}",err);
            4
        }
    };
    HttpServer::new( move || {
        let model_path: &str = &env::var("MODEL_PATH").unwrap();
        let service: &str = &env::var("SERVICE").unwrap();
        let model: BertEmbeddingModel = BertEmbeddingModel::new_from_file(model_path).unwrap();
        App::new()
            .service(
                web::scope(&format!("/{}",service))
                    .app_data(web::Data::new(model))
                    .service(encode)
                    .service(ping)
            )
            .wrap(Logger::default())
    })
    .workers(num_web_workers)
    .bind(("0.0.0.0",5000))?
    .run()
    .await
}