use anyhow;
use log::debug;
use std::fs;
use std::path::Path;
use usearch::{new_index, Index, IndexOptions};

pub struct VectorIndex {
    pub index: Option<Index>,
    path: String,
}

impl VectorIndex {
    pub fn new(path: String, overwrite: bool) -> anyhow::Result<Self> {
        debug!("creating new VectorIndex instance");
        let index_dir = Path::new(&path);
        if overwrite && index_dir.exists() {
            debug!("index already exists, overwriting");
            fs::remove_dir_all(path.as_str())?;
        }

        fs::create_dir_all(path.as_str())?;

        Ok(VectorIndex {
            index: None,
            path: path,
        })
    }

    pub fn with_options(
        &mut self,
        options: &IndexOptions,
        capacity: usize,
    ) -> anyhow::Result<&Self> {
        let index = new_index(options).unwrap();
        index.reserve(capacity).unwrap();
        self.index = Some(index);
        Ok(self)
    }

    pub fn save(&self) -> anyhow::Result<()> {
        let index = self.index.as_ref().unwrap();
        let index_dir = Path::new(&self.path);
        let index_path = index_dir.join("index.bin");

        index.save(index_path.to_str().unwrap()).unwrap();
        Ok(())
    }

    pub fn add(&self, keys: &Vec<u64>, vectors: *const f32) -> anyhow::Result<()> {
        let index = self.index.as_ref().unwrap();
        keys.iter().enumerate().for_each(|(i, _key)| {
            let vector_offset = unsafe { vectors.add(i * 384) };
            let vector: &[f32] = unsafe { std::slice::from_raw_parts(vector_offset, 384) };
            index.add(keys[i], vector).unwrap();
        });

        Ok(())
    }
}
