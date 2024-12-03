use crate::collection::collection_manager::CollectionManager;
use crate::collection::collection_utils::SearchResult;
use actix_web::middleware::Logger;
use actix_web::{web, App, HttpResponse, HttpServer, Responder};
use serde::{Deserialize, Serialize};
use std::time::Instant;
use tokio::sync::RwLock;

#[derive(Serialize)]
struct ErrorResponse {
    status: String,
    message: String,
    time: f64,
}

impl ErrorResponse {
    fn new(message: String, start: Instant) -> Self {
        ErrorResponse {
            status: "error".to_string(),
            message: message,
            time: start.elapsed().as_secs_f64(),
        }
    }
}

#[derive(Serialize)]
struct SuccessResponse<T: Serialize> {
    data: T,
    status: String,
    time: f64,
}

impl<T: Serialize> SuccessResponse<T> {
    fn new(data: T, start: Instant) -> Self {
        SuccessResponse {
            data: data,
            status: "ok".to_string(),
            time: start.elapsed().as_secs_f64(),
        }
    }
}

#[derive(Deserialize)]
struct QueryRequest {
    column_name: String,
    query: String,
    limit: Option<u32>,
}

#[derive(Serialize)]
struct HelthcheckResponse {
    version: String,
    status: String,
    collections: Vec<String>,
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

#[derive(Serialize)]
struct SearchResultsResponse {
    results: Vec<SearchResult>,
}

async fn healthcheck(manager: web::Data<RwLock<CollectionManager>>) -> impl Responder {
    let start = Instant::now();
    let manager_guard = manager.read().await;
    let collections = manager_guard.get_collections().await;
    let response = SuccessResponse::new(
        HelthcheckResponse {
            version: "0.1.0".to_string(),
            status: "ok".to_string(),
            collections: collections,
        },
        start,
    );
    HttpResponse::Ok().json(response)
}

async fn get_collections(manager: web::Data<RwLock<CollectionManager>>) -> impl Responder {
    let start = Instant::now();
    let configs = manager.read().await.get_collection_configs().await;
    let configs_presentable = configs
        .iter()
        .map(|c| CollectionConfigPresentable {
            name: c.name.to_string(),
            index_columns: c.index_columns.to_vec(),
        })
        .collect();
    let response = SuccessResponse::new(
        CollectionsResponse {
            collections: configs_presentable,
        },
        start,
    );
    HttpResponse::Ok().json(response)
}

async fn get_collection(
    collection_name: web::Path<String>,
    manager: web::Data<RwLock<CollectionManager>>,
) -> impl Responder {
    let start = Instant::now();
    let name = collection_name.into_inner();
    let config = manager.read().await.get_collection_config(name).await;
    let response = match config {
        Ok(config) => HttpResponse::Ok().json(SuccessResponse::new(
            CollectionConfigPresentable {
                name: config.name,
                index_columns: config.index_columns,
            },
            start,
        )),
        Err(e) => HttpResponse::NotFound().json(ErrorResponse::new(e.to_string(), start)),
    };

    response
}

async fn search(
    collection_name: web::Path<String>,
    req: web::Json<QueryRequest>,
    manager: web::Data<RwLock<CollectionManager>>,
) -> impl Responder {
    let start = Instant::now();
    let name = collection_name.into_inner();
    let limit = req.limit.unwrap_or(10);
    if limit < 1 || limit > 100 {
        return HttpResponse::BadRequest().json(ErrorResponse::new(
            String::from("Limit should be between 1 and 100"),
            start,
        ));
    }

    let results = manager
        .read()
        .await
        .search(name, req.column_name.clone(), req.query.clone(), limit)
        .await;
    let response = match results {
        Ok(results) => HttpResponse::Ok().json(SuccessResponse::new(
            SearchResultsResponse { results: results },
            start,
        )),
        Err(e) => HttpResponse::NotFound().json(ErrorResponse::new(e.to_string(), start)),
    };

    response
}

pub async fn run_server(
    host: String,
    port: i32,
    collection_name: String,
    token: Option<String>,
) -> std::io::Result<()> {
    let collection_manager = CollectionManager::new(token);
    let _ = collection_manager
        .load_collection(collection_name)
        .await
        .unwrap();
    let shared_manager = web::Data::new(RwLock::new(collection_manager));
    HttpServer::new(move || {
        App::new()
            .app_data(shared_manager.clone())
            .wrap(Logger::new("from %a to %r with %s in %T secs"))
            .route("/", web::get().to(healthcheck))
            .route("/collections", web::get().to(get_collections))
            .route(
                "/collections/{collection_name}",
                web::get().to(get_collection),
            )
            .route(
                "/collections/{collection_name}/search",
                web::post().to(search),
            )
    })
    .bind(format!("{host}:{port}"))?
    .run()
    .await
}
