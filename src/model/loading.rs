/// Weight loading from sharded safetensors files.
///
/// The loading pipeline:
/// 1. Read `model.safetensors.index.json` to get the weight → shard mapping.
/// 2. Memory-map each shard file.
/// 3. For each named tensor, look it up in the appropriate shard's safetensors
///    view, convert from bf16 to f32, and wrap in a [`Tensor`].
///
/// # Assumptions
/// - All weight tensors are stored in bf16 (bfloat16) dtype.
/// - The index file and shard files are in the same directory.

use std::collections::HashMap;
use std::path::{Path, PathBuf};

use memmap2::Mmap;
use safetensors::SafeTensors;
use serde::Deserialize;

use crate::layers::tensor::Bf16Tensor;

/// Partial schema for `model.safetensors.index.json`.
#[derive(Deserialize)]
struct SafetensorsIndex {
    weight_map: HashMap<String, String>,
}

/// A loaded set of safetensor shards, ready for tensor extraction.
pub struct ShardedWeights {
    /// Maps weight name → shard filename.
    weight_map: HashMap<String, String>,
    /// Shard filename → memory-mapped bytes. We keep the `Mmap` alive here so
    /// the `SafeTensors` views remain valid.
    #[allow(dead_code)]
    mmaps: HashMap<String, Mmap>,
    /// Shard filename → raw byte slices (borrowed from mmaps). We need to
    /// store the parsed `SafeTensors` objects, but they borrow from `mmaps`,
    /// so we just re-parse on each tensor access for simplicity (the header
    /// parse is cheap).
    shard_bytes: HashMap<String, *const [u8]>,
    /// Base directory where the shard files live.
    _base_dir: PathBuf,
}

// Safety: the Mmap data doesn't move and we only read it.
unsafe impl Send for ShardedWeights {}
unsafe impl Sync for ShardedWeights {}

impl ShardedWeights {
    /// Load shards from a model directory containing `model.safetensors.index.json`.
    pub fn load(model_dir: &str) -> Self {
        let base = Path::new(model_dir);
        let index_path = base.join("model.safetensors.index.json");
        let index_data = std::fs::read_to_string(&index_path)
            .unwrap_or_else(|e| panic!("Failed to read index: {}", e));
        let index: SafetensorsIndex = serde_json::from_str(&index_data)
            .unwrap_or_else(|e| panic!("Failed to parse index JSON: {}", e));

        // Collect unique shard filenames
        let shard_names: Vec<String> = {
            let mut names: Vec<String> = index.weight_map.values().cloned().collect();
            names.sort();
            names.dedup();
            names
        };

        let mut mmaps = HashMap::new();
        let mut shard_bytes = HashMap::new();

        for name in &shard_names {
            let path = base.join(name);
            let file = std::fs::File::open(&path)
                .unwrap_or_else(|e| panic!("Failed to open shard {}: {}", path.display(), e));
            let mmap = unsafe { Mmap::map(&file) }
                .unwrap_or_else(|e| panic!("Failed to mmap shard {}: {}", path.display(), e));
            let ptr: *const [u8] = &*mmap as &[u8] as *const [u8];
            mmaps.insert(name.clone(), mmap);
            shard_bytes.insert(name.clone(), ptr);
        }

        Self {
            weight_map: index.weight_map,
            mmaps,
            shard_bytes,
            _base_dir: base.to_path_buf(),
        }
    }

    /// Extract a named tensor as a `Bf16Tensor` (no f32 conversion).
    ///
    /// The raw bf16 data is copied out of the memory-mapped shard into a
    /// `Vec<u16>`.  This keeps memory at ~half compared to f32 storage.
    pub fn tensor(&self, name: &str) -> Bf16Tensor {
        let shard_name = self
            .weight_map
            .get(name)
            .unwrap_or_else(|| panic!("Weight '{}' not found in index", name));

        let bytes_ptr = self.shard_bytes.get(shard_name.as_str())
            .unwrap_or_else(|| panic!("Shard '{}' not loaded", shard_name));

        // Safety: the Mmap is alive for the lifetime of self.
        let bytes: &[u8] = unsafe { &**bytes_ptr };

        let st = SafeTensors::deserialize(bytes)
            .unwrap_or_else(|e| panic!("Failed to parse shard '{}': {}", shard_name, e));

        let view = st
            .tensor(name)
            .unwrap_or_else(|e| panic!("Tensor '{}' not in shard '{}': {}", name, shard_name, e));

        let shape: Vec<usize> = view.shape().to_vec();

        // Copy raw bytes as u16 bf16 bits (no f32 conversion)
        let raw = view.data();
        assert_eq!(
            raw.len() % 2,
            0,
            "bf16 tensor data length must be even"
        );
        let num_elements = raw.len() / 2;
        let data: Vec<u16> = (0..num_elements)
            .map(|i| u16::from_le_bytes([raw[2 * i], raw[2 * i + 1]]))
            .collect();

        Bf16Tensor::new(data, shape)
    }

    /// Extract a named tensor as raw bf16 `Vec<u16>` data (1-D, for small
    /// parameters like LayerNorm weights that will be converted to f32 by
    /// the caller).
    pub fn tensor_bf16_raw(&self, name: &str) -> Vec<u16> {
        let t = self.tensor(name);
        t.data
    }
}
