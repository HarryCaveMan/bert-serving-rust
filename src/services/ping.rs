use actix_web::{get,HttpResponse};

#[get("/ping")]
async fn ping() -> HttpResponse {
    HttpResponse::Ok().body("Ready!!")
}