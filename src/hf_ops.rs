use crate::collection::collection_utils::home_dir;
use hf_hub::api::sync::ApiBuilder;
use std::fs;

pub fn download_model(
    model_path: String,
    variant: String,
    token: Option<String>,
) -> anyhow::Result<(String, String)> {
    // Build the Hugging Face API instance
    let cache_dir = home_dir().join("models").to_path_buf();
    let api = ApiBuilder::new()
        .with_token(token)
        .with_cache_dir(cache_dir)
        .build()?;
    let model_path = model_path.replace("hf://", "");
    let repo = api.model(model_path);
    let config_path = repo.get("metadata.json")?;

    // Read the metadata.json file
    let config_content = fs::read_to_string(config_path)?;
    let config: serde_json::Value = serde_json::from_str(&config_content)?;

    // Parse the "letsearch_version" and "variants"
    let version = config["letsearch_version"].as_i64().ok_or_else(|| {
        anyhow::anyhow!("This is probably not a letsearch-compatible model. Check it out")
    })?;
    assert_eq!(version, 1);

    let variants = config["variants"]
        .as_array()
        .ok_or_else(|| anyhow::anyhow!("This is probably not a letsearch model. check it out"))?;

    // Check if the requested variant exists
    let variant_info = variants
        .iter()
        .find(|v| v["variant"] == variant)
        .ok_or_else(|| anyhow::anyhow!("Variant not found in config"))?;

    // Download the ONNX model for the specified variant
    let model_file = match variant_info["path"].as_str() {
        Some(model_path) => repo.get(model_path)?,
        _ => unreachable!("unreachable"),
    };

    if let Some(required_files) = config["required_files"].as_array() {
        for file in required_files {
            repo.get(file.as_str().unwrap())?;
        }
    }

    let model_dir = model_file.parent().unwrap().to_str().unwrap().to_string();
    let model_file = model_file
        .file_name()
        .unwrap()
        .to_str()
        .unwrap()
        .to_string();

    Ok((model_dir, model_file))
}
