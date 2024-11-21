use actix_web::{web, App, HttpResponse, HttpServer, Responder};
use log::info;
use serde::{Deserialize, Serialize};

#[derive(Deserialize)]
struct QueryRequest {
    query: String,
    count: Option<i32>,
}

#[derive(Serialize)]
struct HelthcheckResponse {
    version: String,
    status: String,
}

async fn healthcheck() -> impl Responder {
    let response = HelthcheckResponse {
        version: "0.1.0".to_string(),
        status: "ok".to_string(),
    };
    HttpResponse::Ok().json(response)
}

async fn search(req: web::Json<QueryRequest>) -> impl Responder {
    let query = &req.query;
    let count = req.count.unwrap_or(10);
    info!("got {query}");
    info!("got count: {count}");
    HttpResponse::Ok().body("ok")
}

pub async fn run_server(host: String, port: i32) -> std::io::Result<()> {
    HttpServer::new(|| {
        App::new()
            .route("/", web::get().to(healthcheck))
            .route("/search", web::post().to(search))
    })
    .bind(format!("{host}:{port}"))?
    .run()
    .await
}
