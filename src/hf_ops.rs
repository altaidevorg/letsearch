use crate::collection::collection_utils::home_dir;
use anyhow;
use futures::StreamExt;
use indicatif::{ProgressBar, ProgressStyle};
use reqwest;
use reqwest::header::CONTENT_LENGTH;
use reqwest::header::{HeaderValue, AUTHORIZATION};
use serde::{Deserialize, Serialize};
use std::fs::{self, File};
use std::io::Write;
use std::path::PathBuf;
use std::time::Duration;

#[derive(Deserialize, Debug)]
#[allow(non_snake_case)]
#[allow(dead_code)]
pub struct Model {
    #[serde(rename = "_id")]
    pub id: String,
    pub likes: Option<u64>,
    pub private: bool,
    pub downloads: Option<u64>,
    pub tags: Option<Vec<String>>,
    pub modelId: String,
}

#[derive(Deserialize, Debug)]
#[allow(non_snake_case)]
#[allow(dead_code)]
pub struct ModelInfo {
    pub modelId: Option<String>,
    pub sha: Option<String>,
    pub lastModified: Option<String>,
    pub tags: Option<Vec<String>>,
    pub pipeline_tag: Option<String>,
    pub siblings: Option<Vec<RepoFile>>,
    pub private: bool,
    pub author: Option<String>,
    pub config: Option<serde_json::Value>,
    pub securityStatus: Option<serde_json::Value>,
}

#[derive(Deserialize, Serialize, Debug)]
#[allow(non_snake_case)]
#[allow(dead_code)]
pub struct RepoFile {
    pub rfilename: String,
    pub size: Option<u64>,
    pub blobId: Option<String>,
    pub lfs: Option<BlobLfsInfo>,
}

#[derive(Deserialize, Serialize, Debug)]
#[allow(dead_code)]
pub struct BlobLfsInfo {
    pub size: Option<u64>,
    pub sha256: Option<String>,
    pub pointer_size: Option<u64>,
}

#[allow(dead_code)]
pub async fn get_model_info(repo_id: &str, files_metadata: bool) -> anyhow::Result<ModelInfo> {
    let metadata_param = if files_metadata { "?blobs=true" } else { "" };
    let url = format!(
        "https://huggingface.co/api/models/{}{}",
        repo_id, metadata_param
    );
    let client = reqwest::Client::builder().build()?;
    let response = client.get(&url).send().await?;
    if !response.status().is_success() {
        return Err(anyhow::anyhow!(
            "Failed to fetch model info: {}",
            response.status()
        ));
    }
    let model_info: ModelInfo = response.json().await?;
    Ok(model_info)
}

async fn get_models(filter: &str, token: Option<String>) -> anyhow::Result<Vec<Model>> {
    let url = format!("https://huggingface.co/api/models?filter={}", filter);
    let client = reqwest::Client::builder().build()?;
    let response = match token.as_ref() {
        Some(token) => client.get(&url).header(
            AUTHORIZATION,
            HeaderValue::from_str(format!("BEARER {token}").as_str()).unwrap(),
        ),
        None => client.get(&url),
    }
    .send()
    .await?;

    if !response.status().is_success() {
        return Err(anyhow::anyhow!("Failed to list models: {}", response.status()).into());
    }

    let models: Vec<Model> = response.json().await?;
    Ok(models)
}

async fn download_file(
    repo_id: &str,
    file_name: &str,
    destination_dir: PathBuf,
    token: Option<String>,
) -> anyhow::Result<String> {
    if !destination_dir.exists() {
        fs::create_dir_all(destination_dir.clone())?;
    }

    let destination_path = destination_dir.join(file_name);
    if destination_path.exists() {
        return Ok(destination_path.to_string_lossy().to_string());
    }

    let url = format!(
        "https://huggingface.co/{}/resolve/main/{}",
        repo_id, file_name
    );
    let client = reqwest::Client::builder().build()?;

    let response = match token.as_ref() {
        Some(token) => client.get(&url).header(
            AUTHORIZATION,
            HeaderValue::from_str(format!("BEARER {token}").as_str()).unwrap(),
        ),
        None => client.get(&url),
    }
    .send()
    .await?;

    if !response.status().is_success() {
        return Err(anyhow::anyhow!(
            "Failed to download file: {}",
            response.status()
        ));
    }

    let total_size = response
        .headers()
        .get(CONTENT_LENGTH)
        .and_then(|val| val.to_str().ok()?.parse::<u64>().ok())
        .unwrap_or(0);
    let mut file = File::create(&destination_path)?;

    // Set up the progress bar
    let progress_bar = ProgressBar::new(total_size);
    progress_bar.set_style(
        ProgressStyle::with_template("[{bar:40.cyan/blue}] {bytes}/{total_bytes} ({eta})")
            .unwrap()
            .progress_chars("#>-"),
    );

    let mut downloaded: u64 = 0;

    let mut source = response.bytes_stream();
    while let Some(Ok(chunk)) = source.next().await {
        let bytes_read = chunk.len();
        if bytes_read == 0 {
            break;
        }
        file.write_all(&chunk[..bytes_read])?;
        downloaded += bytes_read as u64;
        progress_bar.set_position(downloaded);
    }

    progress_bar.finish_with_message("Download complete");
    Ok(destination_path.to_string_lossy().to_string())
}

pub async fn download_model(
    model_path: String,
    variant: String,
    token: Option<String>,
) -> anyhow::Result<(String, String)> {
    let cache_dir = home_dir().join("models");
    let repo_id = model_path.replace("hf://", "").to_string();
    let (username, repo_name) = repo_id.split_once("/").ok_or_else(|| {
        anyhow::anyhow!("This is probabably not a proper HuggingFace path. Check it out")
    })?;
    let destination_dir = cache_dir.join(username).join(repo_name);

    let config_path = download_file(
        repo_id.as_str(),
        "metadata.json",
        destination_dir.clone(),
        token.clone(),
    )
    .await?;

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
    let local_model_path = match variant_info["path"].as_str() {
        Some(model_file) => PathBuf::from(
            download_file(
                &repo_id.as_str(),
                model_file,
                destination_dir.clone(),
                token.clone(),
            )
            .await?,
        ),
        None => unreachable!("unreachable"), // we already varified it's a letsearch model, so there shouldn't be a variant without a path key
    };

    if let Some(required_files) = config["required_files"].as_array() {
        for file_name in required_files {
            download_file(
                repo_id.as_str(),
                file_name.as_str().unwrap(),
                destination_dir.clone(),
                token.clone(),
            )
            .await?;
        }
    }

    let model_dir = local_model_path
        .parent()
        .unwrap()
        .to_str()
        .unwrap()
        .to_string();
    let model_file = local_model_path
        .file_name()
        .unwrap()
        .to_str()
        .unwrap()
        .to_string();

    Ok((model_dir, model_file))
}

pub async fn list_models(token: Option<String>) -> anyhow::Result<()> {
    // Create an indefinite spinner progress bar
    let progress_bar = ProgressBar::new_spinner();
    progress_bar.set_style(
        ProgressStyle::default_spinner()
            .template("{spinner:.green} {msg}")
            .expect("Failed to set template")
            .tick_strings(&["⠋", "⠙", "⠹", "⠸", "⠼", "⠴", "⠦", "⠧", "⠇", "⠏"]),
    );
    progress_bar.enable_steady_tick(Duration::from_millis(100));
    progress_bar.set_message("Listing models...");

    let mut models = get_models("letsearch", token.clone()).await?;
    if models.is_empty() {
        progress_bar.finish_and_clear();
        println!("No letsearch-compatible models found on HuggingFace Hub :(");
        println!("Maybe trying to convert your own?");
        return Ok(());
    } else {
        let count = models.len();
        progress_bar.finish_with_message(format!("{} model(s) found!", count));

        println!("===============");
        models.sort_by(|a, b| b.downloads.cmp(&a.downloads));
        for model in models {
            println!("     hf://{}", model.modelId);
        }
    }
    println!("");
    println!("If you cannot see a private model of yours, try using `--hf-token` argument or setting `HF_TOKEN` as an environment variable.");

    Ok(())
}

#[cfg(test)]
mod tests {
    use crate::hf_ops::{download_model, get_model_info, list_models};
    use std::env::temp_dir;
    
    use std::path::PathBuf;

    #[tokio::test]
    async fn test_download_file() {
        let tmp_dir = temp_dir().join("letsearch_models");
        let repo_id = "mys/minilm";
        let file_name = "metadata.json";
        let downloaded_file = super::download_file(repo_id, file_name, tmp_dir, None)
            .await
            .unwrap();
        assert!(PathBuf::from(downloaded_file).exists());
    }

    #[tokio::test]
    async fn test_download_model() {
        let model_path = String::from("hf://mys/minilm");
        let variant = String::from("i8");

        let (model_dir, model_file) = download_model(model_path, variant, None).await.unwrap();

        let model_path = PathBuf::from(&model_dir).join(&model_file);
        assert!(model_path.exists());

        // Clean up
        //fs::remove_dir_all(PathBuf::from(&model_dir)).unwrap();
    }

    #[tokio::test]
    async fn test_get_model_info() {
        let repo_id = "mys/minilm";
        let model_info = get_model_info(repo_id, false).await.unwrap();
        assert!(model_info.modelId.is_some());
    }

    #[tokio::test]
    async fn test_get_models() {
        let models = super::get_models("letsearch", None).await.unwrap();
        assert!(!models.is_empty()); // Assuming there's at least one "letsearch" model
    }

    #[tokio::test]
    async fn test_list_models() {
        // This function primarily prints to stdout, so we'll just check if it completes without error.
        list_models(None).await.unwrap();
    }
}
