use anyhow::Error;
use async_trait::async_trait;

#[async_trait]
pub trait ModelTrait {
    async fn predict(&self, texts: Vec<&str>) -> Result<String, String>;
    async fn load_model(&mut self, model_path: &str) -> Result<(), Error>;
    async fn unload_model(&self) -> Result<(), String>;
}
