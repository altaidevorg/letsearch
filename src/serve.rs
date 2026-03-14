use crate::actors::collection_actor::GetConfig;
use crate::actors::collection_manager_actor::{
    CollectionManagerActor, GetAllCollectionConfigs, GetCollectionAddr,
    LoadCollection, SearchCollection
};
use crate::actors::model_actor::ModelManagerActor;
use crate::collection::collection_utils::SearchResult;
use actix::{Actor, Addr};
use actix_web::middleware::Logger;
use actix_web::{web, App, HttpResponse, HttpServer, Responder};
use serde::{Deserialize, Serialize};
use std::time::Instant;

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

async fn healthcheck() -> impl Responder {
    let start = Instant::now();
    let response = SuccessResponse::new(
        HelthcheckResponse {
            version: env!("CARGO_PKG_VERSION").to_string(),
            status: "ok".to_string(),
        },
        start,
    );
    HttpResponse::Ok().json(response)
}

async fn get_collections(manager: web::Data<Addr<CollectionManagerActor>>) -> impl Responder {
    let start = Instant::now();
    let result = manager.send(GetAllCollectionConfigs).await;
    match result {
        Ok(Ok(configs)) => {
            let configs_presentable = configs
                .into_iter()
                .map(|c| CollectionConfigPresentable {
                    name: c.name,
                    index_columns: c.index_columns,
                })
                .collect();
            HttpResponse::Ok().json(SuccessResponse::new(
                CollectionsResponse {
                    collections: configs_presentable,
                },
                start,
            ))
        }
        _ => HttpResponse::InternalServerError().json(ErrorResponse::new(
            "Failed to retrieve collections".to_string(),
            start,
        )),
    }
}

async fn get_collection(
    collection_name: web::Path<String>,
    manager: web::Data<Addr<CollectionManagerActor>>,
) -> impl Responder {
    let start = Instant::now();
    let name = collection_name.into_inner();
    let result = manager.send(GetCollectionAddr { name }).await;

    match result {
        Ok(Ok(collection_addr)) => {
            let config_result = collection_addr.send(GetConfig).await;
            match config_result {
                Ok(Ok(config)) => HttpResponse::Ok().json(SuccessResponse::new(
                    CollectionConfigPresentable {
                        name: config.name,
                        index_columns: config.index_columns,
                    },
                    start,
                )),
                _ => HttpResponse::InternalServerError().json(ErrorResponse::new(
                    "Failed to get collection config".to_string(),
                    start,
                )),
            }
        }
        Ok(Err(e)) => HttpResponse::NotFound().json(ErrorResponse::new(e.to_string(), start)),
        _ => HttpResponse::InternalServerError().json(ErrorResponse::new(
            "Failed to find collection".to_string(),
            start,
        )),
    }
}

async fn search(
    collection_name: web::Path<String>,
    req: web::Json<QueryRequest>,
    manager: web::Data<Addr<CollectionManagerActor>>,
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

    let search_result = manager.send(SearchCollection {
        collection_name: name.clone(),
        column: req.column_name.clone(),
        query: req.query.clone(),
        limit,
    }).await;

    match search_result {
        Ok(Ok(results)) => HttpResponse::Ok().json(SuccessResponse::new(
            SearchResultsResponse { results },
            start,
        )),
        Ok(Err(e)) => HttpResponse::NotFound().json(ErrorResponse::new(e.to_string(), start)),
        _ => HttpResponse::InternalServerError().json(ErrorResponse::new(
            "Search request to manager failed".to_string(),
            start,
        )),
    }
}

pub async fn run_server(
    host: String,
    port: i32,
    collection_name: String,
    token: Option<String>,
) -> std::io::Result<()> {
    let model_manager_addr = ModelManagerActor::new().start();
    let collection_manager_addr =
        CollectionManagerActor::new(token, model_manager_addr.clone()).start();

    let load_result = collection_manager_addr
        .send(LoadCollection {
            name: collection_name,
        })
        .await;

    if let Err(e) = load_result
        .map_err(|e| anyhow::anyhow!(e))
        .and_then(|r| r.map_err(|e| anyhow::anyhow!(e)))
    {
        panic!("Failed to load initial collection: {:?}", e);
    }

    let shared_manager_addr = web::Data::new(collection_manager_addr);

    HttpServer::new(move || {
        App::new()
            .app_data(shared_manager_addr.clone())
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
