//! Training examples.

/// One tokenized example. `train[i]` says whether token `i` counts toward the loss.
#[derive(Debug, Clone)]
pub struct Example {
    pub tokens: Vec<u32>,
    pub train: Vec<bool>,
}
