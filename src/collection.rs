use anyhow;
use duckdb::Connection;
use log::debug;
use std::fs;
use std::path::Path;

pub struct Collection {
    name: String,
    conn: Connection,
}

impl Collection {
    pub fn new(name: String, overwrite: bool) -> anyhow::Result<Self> {
        debug!("creating new Collection instance");
        let collection_dir = Path::new(&name);
        if overwrite && collection_dir.exists() {
            debug!("Collection already exists, overwriting");
            fs::remove_dir_all(name.as_str())?;
        }

        fs::create_dir_all(name.as_str())?;
        let db_path = collection_dir.join("data.db");
        let conn = Connection::open(db_path)?;
        debug!("Connection opened to DB");

        Ok(Collection {
            name: name,
            conn: conn,
        })
    }

    pub fn import_jsonl(&self, jsonl_path: &str) -> anyhow::Result<()> {
        self.conn.execute_batch(
            format!(
                "CREATE TABLE {} AS SELECT * FROM read_json_auto('{}');",
                &self.name, jsonl_path
            )
            .as_str(),
        )?;
        debug!("JSONL file imported");

        Ok(())
    }
}
