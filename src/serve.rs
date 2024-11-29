use crate::collection::collection::Collection;
use crate::collection::collection_utils::CollectionConfig;
use actix_web::{web, App, HttpResponse, HttpServer, Responder};
use log::info;
use serde::{Deserialize, Serialize};
use std::sync::Mutex;

#[derive(Deserialize)]
struct QueryRequest {
    query: String,
    count: Option<i32>,
}

#[derive(Serialize)]
struct HelthcheckResponse {
    version: String,
    status: String,
    collections: Vec<CollectionConfig>,
}

async fn healthcheck(collection: web::Data<Mutex<Collection>>) -> impl Responder {
    let collection = collection.lock().unwrap();
    let response = HelthcheckResponse {
        version: "0.1.0".to_string(),
        status: "ok".to_string(),
        collections: vec![collection.config()],
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

pub async fn run_server(host: String, port: i32, collection_name: String) -> std::io::Result<()> {
    let collection = Collection::from(collection_name).unwrap();
    let shared_collection = web::Data::new(Mutex::new(collection));
    HttpServer::new(move || {
        App::new()
            .app_data(shared_collection.clone())
            .route("/", web::get().to(healthcheck))
            .route("/search", web::post().to(search))
    })
    .bind(format!("{host}:{port}"))?
    .run()
    .await
}
