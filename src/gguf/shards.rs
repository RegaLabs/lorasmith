//! Reads tensors out of `.safetensors` files one at a time, without loading whole files.

use std::fs::File;
use std::io::{Read, Seek, SeekFrom};
use std::path::{Path, PathBuf};

use anyhow::{Context, Result, bail};
use burn::tensor::{DType, TensorData};
use serde_json::Value as Json;

/// Where one tensor's data comes from.
pub enum Source {
    File {
        path: PathBuf,
        dtype: DType,
        start: u64,
        end: u64,
    },
    Values(Vec<f32>),
}

/// Every tensor in the folder's `.safetensors` files, read from their headers only.
pub fn tensors(dir: &Path) -> Result<Vec<(String, Vec<usize>, Source)>> {
    let mut shards: Vec<PathBuf> = std::fs::read_dir(dir)?
        .filter_map(|e| Some(e.ok()?.path()))
        .filter(|p| p.extension().is_some_and(|e| e == "safetensors"))
        .collect();
    shards.sort();
    let mut tensors = Vec::new();
    for path in shards {
        let mut file =
            File::open(&path).with_context(|| format!("can't open {}", path.display()))?;
        let mut len = [0u8; 8];
        file.read_exact(&mut len)?;
        let mut header = vec![0u8; u64::from_le_bytes(len) as usize];
        file.read_exact(&mut header)?;
        let header: Json = serde_json::from_slice(&header)
            .with_context(|| format!("{} is damaged", path.display()))?;
        let data_start = 8 + u64::from_le_bytes(len);
        for (name, entry) in header.as_object().context("bad safetensors header")? {
            if name == "__metadata__" {
                continue;
            }
            let dtype = match entry["dtype"].as_str() {
                Some("F32") => DType::F32,
                Some("F16") => DType::F16,
                Some("BF16") => DType::BF16,
                other => {
                    bail!("{name} is stored as {other:?}; only F32, F16 and BF16 are supported")
                }
            };
            let shape = entry["shape"]
                .as_array()
                .context("bad shape")?
                .iter()
                .filter_map(Json::as_u64)
                .map(|d| d as usize)
                .collect();
            let offsets: Vec<u64> = entry["data_offsets"]
                .as_array()
                .context("bad offsets")?
                .iter()
                .filter_map(Json::as_u64)
                .collect();
            let source = Source::File {
                path: path.clone(),
                dtype,
                start: data_start + offsets[0],
                end: data_start + offsets[1],
            };
            tensors.push((name.clone(), shape, source));
        }
    }
    tensors.sort_by(|a, b| a.0.cmp(&b.0));
    Ok(tensors)
}

/// The tensor's values as f32, whatever precision the file stores.
pub fn read(source: &Source) -> Result<Vec<f32>> {
    let (path, dtype, start, end) = match source {
        Source::Values(values) => return Ok(values.clone()),
        Source::File {
            path,
            dtype,
            start,
            end,
        } => (path, *dtype, *start, *end),
    };
    let mut file = File::open(path)?;
    file.seek(SeekFrom::Start(start))?;
    let mut bytes = vec![0u8; (end - start) as usize];
    file.read_exact(&mut bytes)?;
    let count = bytes.len() / dtype.size();
    let data = TensorData::from_bytes_vec(bytes, [count], dtype).convert_dtype(DType::F32);
    data.try_to_vec::<f32>()
        .map_err(|e| anyhow::anyhow!("{e:?}"))
}
