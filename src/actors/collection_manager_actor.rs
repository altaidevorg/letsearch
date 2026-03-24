use crate::actors::collection_actor::{CollectionActor, GetConfig, Search as SearchMsg};
use crate::actors::model_actor::{LoadModel, ModelManagerActor};
use crate::collection::collection_utils::{CollectionConfig, SearchResult};
use crate::error::ProjectError;
use actix::prelude::*;
use std::collections::HashMap;

// ---- Actor Definition ----
pub struct CollectionManagerActor {
    collections: HashMap<String, Addr<CollectionActor>>,
    model_manager: Addr<ModelManagerActor>,
    model_lookup: HashMap<(String, String), u32>,
    hf_token: Option<String>,
    gemini_api_key: Option<String>,
}

impl CollectionManagerActor {
    pub fn new(
        hf_token: Option<String>,
        model_manager: Addr<ModelManagerActor>,
        gemini_api_key: Option<String>,
    ) -> Self {
        Self {
            collections: HashMap::new(),
            model_manager,
            model_lookup: HashMap::new(),
            hf_token,
            gemini_api_key,
        }
    }
}

impl Actor for CollectionManagerActor {
    type Context = Context<Self>;
}

// ---- Message Definitions ----
#[derive(Message)]
#[rtype(result = "Result<Addr<CollectionActor>, ProjectError>")]
pub struct CreateCollection {
    pub config: CollectionConfig,
    pub overwrite: bool,
}

#[derive(Message)]
#[rtype(result = "Result<Addr<CollectionActor>, ProjectError>")]
pub struct LoadCollection {
    pub name: String,
}

#[derive(Message)]
#[rtype(result = "Result<Addr<CollectionActor>, ProjectError>")]
pub struct GetCollectionAddr {
    pub name: String,
}

#[derive(Message)]
#[rtype(result = "Result<Vec<CollectionConfig>, ProjectError>")]
pub struct GetAllCollectionConfigs;

#[derive(Message)]
#[rtype(result = "()")]
struct UpdateCollection {
    name: String,
    addr: Addr<CollectionActor>,
    model_key: (String, String),
    model_id: u32,
}

#[derive(Message)]
#[rtype(result = "Result<u32, ProjectError>")]
pub struct GetModelIdForCollection {
    pub name: String,
}

#[derive(Message)]
#[rtype(result = "Result<Vec<SearchResult>, ProjectError>")]
pub struct SearchCollection {
    pub collection_name: String,
    pub column: String,
    pub query: String,
    pub limit: u32,
}

// ---- Message Handlers ----
impl Handler<UpdateCollection> for CollectionManagerActor {
    type Result = ();

    fn handle(&mut self, msg: UpdateCollection, _ctx: &mut Context<Self>) -> Self::Result {
        self.collections.insert(msg.name, msg.addr);
        self.model_lookup.insert(msg.model_key, msg.model_id);
    }
}

impl Handler<GetCollectionAddr> for CollectionManagerActor {
    type Result = Result<Addr<CollectionActor>, ProjectError>;

    fn handle(&mut self, msg: GetCollectionAddr, _ctx: &mut Context<Self>) -> Self::Result {
        self.collections
            .get(&msg.name)
            .cloned()
            .ok_or_else(|| ProjectError::CollectionNotFound(msg.name))
    }
}

impl Handler<GetAllCollectionConfigs> for CollectionManagerActor {
    type Result = ResponseFuture<Result<Vec<CollectionConfig>, ProjectError>>;

    fn handle(&mut self, _msg: GetAllCollectionConfigs, _ctx: &mut Context<Self>) -> Self::Result {
        let futures: Vec<_> = self
            .collections
            .values()
            .map(|addr| addr.send(GetConfig))
            .collect();

        Box::pin(async move {
            let results = futures::future::join_all(futures).await;
            let configs: Result<Vec<CollectionConfig>, _> = results
                .into_iter()
                .map(|res| match res {
                    Ok(Ok(config)) => Ok(config),
                    Ok(Err(e)) => Err(e),
                    Err(e) => Err(ProjectError::Mailbox(e)),
                })
                .collect();
            configs
        })
    }
}

impl Handler<CreateCollection> for CollectionManagerActor {
    type Result = ResponseFuture<Result<Addr<CollectionActor>, ProjectError>>;

    fn handle(&mut self, msg: CreateCollection, ctx: &mut Context<Self>) -> Self::Result {
        let collection_name = msg.config.name.clone();
        if self.collections.contains_key(&collection_name) && !msg.overwrite {
            return Box::pin(async move {
                Err(ProjectError::Anyhow(anyhow::anyhow!(
                    "Collection '{}' already exists.",
                    collection_name
                )))
            });
        }

        let model_manager = self.model_manager.clone();
        let model_key = (
            msg.config.model_name.clone(),
            msg.config.model_variant.clone(),
        );
        let hf_token = self.hf_token.clone();
        let gemini_api_key = self.gemini_api_key.clone();
        let self_addr = ctx.address();

        Box::pin(async move {
            let model_id = model_manager
                .send(LoadModel {
                    path: model_key.0.clone(),
                    variant: model_key.1.clone(),
                    token: hf_token,
                    gemini_api_key,
                })
                .await??;

            let collection_actor = CollectionActor::new(msg.config, model_manager);
            let collection_addr = collection_actor.start();

            self_addr.do_send(UpdateCollection {
                name: collection_name,
                addr: collection_addr.clone(),
                model_key,
                model_id,
            });

            Ok(collection_addr)
        })
    }
}

impl Handler<LoadCollection> for CollectionManagerActor {
    type Result = ResponseFuture<Result<Addr<CollectionActor>, ProjectError>>;

    fn handle(&mut self, msg: LoadCollection, ctx: &mut Context<Self>) -> Self::Result {
        if let Some(addr) = self.collections.get(&msg.name).cloned() {
            return Box::pin(async move { Ok(addr) });
        }

        let model_manager = self.model_manager.clone();
        let name = msg.name.clone();
        let hf_token = self.hf_token.clone();
        let gemini_api_key = self.gemini_api_key.clone();
        let self_addr = ctx.address();

        Box::pin(async move {
            let config = CollectionConfig::from_file(&name)?;
            let model_key = (config.model_name.clone(), config.model_variant.clone());
            let model_id = model_manager
                .send(LoadModel {
                    path: model_key.0.clone(),
                    variant: model_key.1.clone(),
                    token: hf_token,
                    gemini_api_key,
                })
                .await??;

            let actor = CollectionActor::new(config, model_manager);
            let collection_addr = actor.start();

            self_addr.do_send(UpdateCollection {
                name,
                addr: collection_addr.clone(),
                model_key,
                model_id,
            });

            Ok(collection_addr)
        })
    }
}

impl Handler<GetModelIdForCollection> for CollectionManagerActor {
    type Result = ResponseFuture<Result<u32, ProjectError>>;

    fn handle(&mut self, msg: GetModelIdForCollection, _ctx: &mut Context<Self>) -> Self::Result {
        let collection_addr = match self.collections.get(&msg.name) {
            Some(addr) => addr.clone(),
            None => {
                return Box::pin(async move { Err(ProjectError::CollectionNotFound(msg.name)) });
            }
        };

        let model_lookup = self.model_lookup.clone();

        Box::pin(async move {
            let config = collection_addr.send(GetConfig).await??;
            let model_key = (config.model_name, config.model_variant);
            model_lookup
                .get(&model_key)
                .copied()
                .ok_or_else(|| ProjectError::ModelNotFound(0)) // 0 is a placeholder
        })
    }
}

impl Handler<SearchCollection> for CollectionManagerActor {
    type Result = ResponseFuture<Result<Vec<SearchResult>, ProjectError>>;

    fn handle(&mut self, msg: SearchCollection, _ctx: &mut Context<Self>) -> Self::Result {
        let collection_addr = match self.collections.get(&msg.collection_name) {
            Some(addr) => addr.clone(),
            None => {
                return Box::pin(async move {
                    Err(ProjectError::CollectionNotFound(msg.collection_name))
                });
            }
        };

        let model_lookup = self.model_lookup.clone();

        Box::pin(async move {
            let config = collection_addr.send(GetConfig).await??;
            let model_key = (config.model_name, config.model_variant);
            let model_id = model_lookup
                .get(&model_key)
                .copied()
                .ok_or_else(|| ProjectError::ModelNotFound(0))?;

            let search_results = collection_addr
                .send(SearchMsg {
                    column: msg.column,
                    query: msg.query,
                    limit: msg.limit,
                    model_id,
                })
                .await??;

            Ok(search_results)
        })
    }
}
