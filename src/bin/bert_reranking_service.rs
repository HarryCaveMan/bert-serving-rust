use actix_web::{web,App,HttpServer};
use actix_web::middleware::{Logger};
use log::{error,info};
use std::{env};
use bert_serving_rust::bert::bert_reranking::model::{BertRerankingModel};
use bert_serving_rust::services::bert_reranking_service::{predict};
use bert_serving_rust::services::ping::{ping};

#[actix_web::main]
async fn main() -> std::io::Result<()> {
    env_logger::init();
    let num_web_workers: usize = match env::var("NUM_WEB_WORKERS") {
        Ok(val) => val.parse::<usize>().unwrap(),
        Err(err) => {
            error!("{:?}",err);
            4
        }
    };
    HttpServer::new( move || {
        let model_path: &str = &env::var("MODEL_PATH").unwrap();
        let service: &str = &env::var("SERVICE").unwrap();
        info!("starting server for {:?}",service);
        info!("Model Path {:?}",model_path);
        let model: BertRerankingModel = BertRerankingModel::new_from_file(model_path).unwrap();
        App::new()
            .service(
                web::scope(&format!("/{}",service))
                    .app_data(web::Data::new(model))
                    .service(predict)
                    .service(ping)
            )
            .wrap(Logger::default())
    })
    .workers(num_web_workers)
    .bind(("0.0.0.0",5000))?
    .run()
    .await
}