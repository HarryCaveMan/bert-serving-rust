use actix_web::{post,web,HttpResponse};
use serde::{Serialize,Deserialize};
use log::{debug};
use std::time::{Instant};
use super::bert::bert_reranking::model::{BertRerankingModel,RankedResults};


#[derive(Serialize,Deserialize)]
struct RerankingRequest {
    crid: u32,
    queries: Vec<String>,
    results_set:Vec<Vec<String>>,
    logit_index_thresh: i64
}
#[derive(Serialize,Deserialize)]
struct RerankingResponse {
    crid: u32,
    results: RankedResults
}

#[post("/predict")]
async fn predict(model: web::Data<BertRerankingModel>, req: web::Json<RerankingRequest>) -> HttpResponse {
    let crid: u32 = req.crid;
    debug!("Starting inference...");
    let start: Instant = Instant::now();
    let results: RankedResults = model.predict(&req.queries,&req.results_set,req.logit_index_thresh);
    debug!("Done! Took {:?}ms",start.elapsed().as_millis());
    debug!("Ranked Results:\n{:?}",results);
    HttpResponse::Ok().json(RerankingResponse{crid:crid,results:results})
}