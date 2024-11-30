use crate::collection::collection_manager::CollectionManager;
use actix_web::{web, App, HttpResponse, HttpServer, Responder};
use log::info;
use serde::{Deserialize, Serialize};
use tokio::sync::RwLock;

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

#[derive(Serialize)]
struct CollectionConfigPresentable {
    name: String,
    index_columns: Vec<String>,
}

#[derive(Serialize)]
struct CollectionsResponse {
    collections: Vec<CollectionConfigPresentable>,
}

async fn healthcheck() -> impl Responder {
    let response = HelthcheckResponse {
        version: "0.1.0".to_string(),
        status: "ok".to_string(),
    };
    HttpResponse::Ok().json(response)
}

async fn get_collections(manager: web::Data<RwLock<CollectionManager>>) -> impl Responder {
    let configs = manager.read().await.get_collection_configs().await;
    let configs_presentable = configs
        .iter()
        .map(|c| CollectionConfigPresentable {
            name: c.name.to_string(),
            index_columns: c.index_columns.to_vec(),
        })
        .collect();
    let response = CollectionsResponse {
        collections: configs_presentable,
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
    let collection_manager = CollectionManager::new();
    let _ = collection_manager
        .load_collection(collection_name)
        .await
        .unwrap();
    let shared_manager = web::Data::new(RwLock::new(collection_manager));
    HttpServer::new(move || {
        App::new()
            .app_data(shared_manager.clone())
            .route("/", web::get().to(healthcheck))
            .route("/collections", web::get().to(get_collections))
            .route("/search", web::post().to(search))
    })
    .bind(format!("{host}:{port}"))?
    .run()
    .await
}
