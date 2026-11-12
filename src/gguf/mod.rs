//! Writes GGUF, the single-file format llama.cpp, Ollama and LM Studio run on any hardware.

mod model;
mod sentencepiece;
mod vocab;

pub use model::convert;

use std::fs::File;
use std::io::{BufWriter, Write};
use std::path::Path;

use anyhow::{Context, Result, ensure};
use burn::tensor::{DType, TensorData};

const ALIGNMENT: usize = 32;

/// A metadata value. Only the kinds llama.cpp reads for our models.
#[derive(Debug, Clone, PartialEq)]
pub enum Value {
    U32(u32),
    F32(f32),
    Bool(bool),
    Str(String),
    Strs(Vec<String>),
    I32s(Vec<i32>),
    F32s(Vec<f32>),
}

/// Storage type of a tensor's data.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum Kind {
    F32,
    F16,
    /// Blocks of 32 values: an f16 scale, then 32 signed bytes. Half the size of f16 and,
    /// for chat, indistinguishable from it.
    Q8_0,
}

impl Kind {
    fn ggml_type(self) -> u32 {
        match self {
            Kind::F32 => 0,
            Kind::F16 => 1,
            Kind::Q8_0 => 8,
        }
    }

    fn bytes(self, values: usize) -> usize {
        match self {
            Kind::F32 => values * 4,
            Kind::F16 => values * 2,
            Kind::Q8_0 => values / 32 * 34,
        }
    }
}

/// Stores `values` as `kind`, bit for bit as llama.cpp's own converter would.
pub fn encode(kind: Kind, values: Vec<f32>) -> Vec<u8> {
    let count = values.len();
    match kind {
        Kind::F32 => TensorData::new(values, [count]).as_bytes().to_vec(),
        Kind::F16 => TensorData::new(values, [count])
            .convert_dtype(DType::F16)
            .as_bytes()
            .to_vec(),
        Kind::Q8_0 => q8_0(&values),
    }
}

fn q8_0(values: &[f32]) -> Vec<u8> {
    let scales: Vec<f32> = values
        .chunks(32)
        .map(|block| block.iter().fold(0f32, |max, v| max.max(v.abs())) / 127.0)
        .collect();
    let half_scales = TensorData::new(scales.clone(), [scales.len()]).convert_dtype(DType::F16);
    let mut out = Vec::with_capacity(values.len() / 32 * 34);
    for ((block, scale), half) in values
        .chunks(32)
        .zip(&scales)
        .zip(half_scales.as_bytes().chunks(2))
    {
        out.extend_from_slice(half);
        let inverse = if *scale == 0.0 { 0.0 } else { 1.0 / scale };
        // Rust's round() rounds halves away from zero, like ggml's roundf.
        out.extend(block.iter().map(|v| (v * inverse).round() as i8 as u8));
    }
    out
}

/// A tensor's name, its shape in Hugging Face order (outermost first) and how it is stored.
#[derive(Debug, Clone)]
pub struct Info {
    pub name: String,
    pub shape: Vec<usize>,
    pub kind: Kind,
}

impl Info {
    fn bytes(&self) -> usize {
        self.kind.bytes(self.shape.iter().product())
    }
}

/// Writes a GGUF v3 file. `data(i)` must return the little-endian bytes of `tensors[i]`;
/// tensors are fetched one at a time so a large model never sits in memory twice.
pub fn write(
    path: &Path,
    metadata: &[(String, Value)],
    tensors: &[Info],
    mut data: impl FnMut(usize) -> Result<Vec<u8>>,
) -> Result<()> {
    let file = File::create(path).with_context(|| format!("can't create {}", path.display()))?;
    let mut out = BufWriter::new(file);
    out.write_all(b"GGUF")?;
    out.write_all(&3u32.to_le_bytes())?;
    out.write_all(&(tensors.len() as u64).to_le_bytes())?;
    out.write_all(&(metadata.len() as u64).to_le_bytes())?;
    for (key, value) in metadata {
        string(&mut out, key)?;
        self::value(&mut out, value)?;
    }

    let mut offset = 0;
    for info in tensors {
        string(&mut out, &info.name)?;
        out.write_all(&(info.shape.len() as u32).to_le_bytes())?;
        // GGUF lists dimensions innermost first, the reverse of PyTorch.
        for &dim in info.shape.iter().rev() {
            out.write_all(&(dim as u64).to_le_bytes())?;
        }
        out.write_all(&info.kind.ggml_type().to_le_bytes())?;
        out.write_all(&(offset as u64).to_le_bytes())?;
        offset += padded(info.bytes());
    }

    let mut written = 4 + 4 + 8 + 8;
    written += metadata
        .iter()
        .map(|(k, v)| 8 + k.len() + value_len(v))
        .sum::<usize>();
    written += tensors
        .iter()
        .map(|t| 8 + t.name.len() + 4 + 8 * t.shape.len() + 4 + 8)
        .sum::<usize>();
    pad(&mut out, written)?;

    for (i, info) in tensors.iter().enumerate() {
        let bytes = data(i)?;
        ensure!(
            bytes.len() == info.bytes(),
            "{} has the wrong number of bytes",
            info.name
        );
        out.write_all(&bytes)?;
        pad(&mut out, bytes.len())?;
    }
    out.flush()
        .with_context(|| format!("can't write {}", path.display()))
}

fn padded(len: usize) -> usize {
    len.div_ceil(ALIGNMENT) * ALIGNMENT
}

fn pad(out: &mut impl Write, len: usize) -> Result<()> {
    out.write_all(&vec![0u8; padded(len) - len])?;
    Ok(())
}

fn string(out: &mut impl Write, text: &str) -> Result<()> {
    out.write_all(&(text.len() as u64).to_le_bytes())?;
    out.write_all(text.as_bytes())?;
    Ok(())
}

fn value(out: &mut impl Write, value: &Value) -> Result<()> {
    const U32: u32 = 4;
    const I32: u32 = 5;
    const F32: u32 = 6;
    const BOOL: u32 = 7;
    const STRING: u32 = 8;
    const ARRAY: u32 = 9;
    match value {
        Value::U32(v) => {
            out.write_all(&U32.to_le_bytes())?;
            out.write_all(&v.to_le_bytes())?;
        }
        Value::F32(v) => {
            out.write_all(&F32.to_le_bytes())?;
            out.write_all(&v.to_le_bytes())?;
        }
        Value::Bool(v) => {
            out.write_all(&BOOL.to_le_bytes())?;
            out.write_all(&[u8::from(*v)])?;
        }
        Value::Str(v) => {
            out.write_all(&STRING.to_le_bytes())?;
            string(out, v)?;
        }
        Value::Strs(items) => {
            out.write_all(&ARRAY.to_le_bytes())?;
            out.write_all(&STRING.to_le_bytes())?;
            out.write_all(&(items.len() as u64).to_le_bytes())?;
            for item in items {
                string(out, item)?;
            }
        }
        Value::I32s(items) => {
            out.write_all(&ARRAY.to_le_bytes())?;
            out.write_all(&I32.to_le_bytes())?;
            out.write_all(&(items.len() as u64).to_le_bytes())?;
            for item in items {
                out.write_all(&item.to_le_bytes())?;
            }
        }
        Value::F32s(items) => {
            out.write_all(&ARRAY.to_le_bytes())?;
            out.write_all(&F32.to_le_bytes())?;
            out.write_all(&(items.len() as u64).to_le_bytes())?;
            for item in items {
                out.write_all(&item.to_le_bytes())?;
            }
        }
    }
    Ok(())
}

/// Bytes `value` takes after its key, including the type tag.
fn value_len(value: &Value) -> usize {
    4 + match value {
        Value::U32(_) | Value::F32(_) => 4,
        Value::Bool(_) => 1,
        Value::Str(v) => 8 + v.len(),
        Value::Strs(items) => 4 + 8 + items.iter().map(|s| 8 + s.len()).sum::<usize>(),
        Value::I32s(items) => 4 + 8 + 4 * items.len(),
        Value::F32s(items) => 4 + 8 + 4 * items.len(),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn q8_0_matches_ggml_blocks() {
        let values: Vec<f32> = (0..32).map(|v| v as f32 - 8.0).collect();
        let bytes = encode(Kind::Q8_0, values);
        assert_eq!(bytes.len(), 34);
        // Largest magnitude 23 maps to 127; 0.5-style halves round away from zero.
        let scale = 23.0f32 / 127.0;
        let half = TensorData::new(vec![scale], [1]).convert_dtype(DType::F16);
        assert_eq!(&bytes[..2], half.as_bytes());
        assert_eq!(bytes[2] as i8, (-8.0f32 / scale).round() as i8);
        assert_eq!(bytes[33] as i8, 127);
        assert_eq!(encode(Kind::Q8_0, vec![0.0; 32])[2..], [0u8; 32]);
    }

    #[test]
    fn layout_matches_the_spec() {
        let path = std::env::temp_dir().join(format!("lorasmith-{}.gguf", std::process::id()));
        let metadata = [(
            "general.architecture".to_string(),
            Value::Str("llama".into()),
        )];
        let tensors = [Info {
            name: "w".into(),
            shape: vec![2, 3],
            kind: Kind::F32,
        }];
        write(&path, &metadata, &tensors, |_| Ok(vec![7u8; 24])).unwrap();
        let bytes = std::fs::read(&path).unwrap();
        std::fs::remove_file(&path).unwrap();

        assert_eq!(&bytes[..4], b"GGUF");
        assert_eq!(u64::from_le_bytes(bytes[8..16].try_into().unwrap()), 1);
        let info_start = 24 + 8 + 20 + 4 + 8 + 5;
        let dims = &bytes[info_start + 9 + 4..info_start + 9 + 4 + 16];
        assert_eq!(
            dims,
            [3u64.to_le_bytes(), 2u64.to_le_bytes()].concat(),
            "innermost first"
        );
        assert_eq!(bytes.len() % ALIGNMENT, 0);
        assert_eq!(&bytes[bytes.len() - 32..bytes.len() - 8], &[7u8; 24]);
    }
}
