use actix_web::{get,web,App,HttpServer,HttpResponse};
use actix_web::middleware::{Logger};
use log::{info};
use std::{env};

use bert_serving_rust::bert::bert_ner::model::{BertNERModel};
use bert_serving_rust::services::bert_ner_service::{predict};

#[get("/ping")]
async fn ping() -> HttpResponse {
    HttpResponse::Ok().body("Ready!!")
}

#[actix_web::main]
async fn main() -> std::io::Result<()> {
    env_logger::init();
    HttpServer::new( move || {
        let model_path: &str = &env::var("MODEL_PATH").unwrap();
        let service: &str = &env::var("SERVICE").unwrap();
        let model: BertNERModel = BertNERModel::new_from_file(model_path).unwrap();
        info!("Starting service on port 5000...");
        App::new()
            .service(
                web::scope(&format!("/{}",service))
                    .app_data(web::Data::new(model))
                    .service(predict)
                    .service(ping)
            )
            .wrap(Logger::default())
    })
    .workers(4)
    .bind(("0.0.0.0",5000))?
    .run()
    .await
}