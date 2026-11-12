//! Finding hardware to train on.

use std::panic::{self, AssertUnwindSafe};
use std::sync::{Arc, Mutex};

use anyhow::{Result, anyhow};
use burn::tensor::{Device, DeviceKind, DeviceType, Distribution, FloatDType, Tensor};

/// A device with a name a person recognizes.
pub struct Found {
    pub name: String,
    pub device: Device,
    /// The precision models train in on it: f16 where the device computes in it, which
    /// halves the memory the weights take.
    pub dtype: FloatDType,
}

/// Every device worth training on, GPUs first and the CPU last.
pub fn list() -> Vec<Found> {
    let gpus = quietly(|| Device::enumerate(DeviceType::Wgpu).into_vec()).unwrap_or_default();
    let mut found: Vec<Found> = gpus.into_iter().filter_map(gpu).collect();
    found.push(cpu());
    found
}

/// The device training runs on: the strongest GPU wgpu can find, or the CPU if none works.
///
/// `CUBECL_WGPU_DEFAULT_DEVICE` overrides the pick, for example `IntegratedGpu(0)`.
pub fn best() -> Found {
    gpu(Device::wgpu(DeviceKind::DefaultDevice))
        .filter(|found| works(&found.device))
        .unwrap_or_else(cpu)
}

/// Runs a small matmul to prove the device really computes.
pub fn works(device: &Device) -> bool {
    quietly(|| {
        let a = Tensor::<2>::random([256, 256], Distribution::Default, device);
        let sum: f32 = a.clone().matmul(a).sum().into_scalar();
        sum.is_finite()
    })
    .unwrap_or(false)
}

fn gpu(device: Device) -> Option<Found> {
    let identity = quietly(|| device.capabilities())
        .flatten()?
        .properties
        .identity;
    // Software rasterizers like llvmpipe pose as GPUs but lose badly to the CPU backend.
    identity.physical.as_ref()?;
    let half = quietly(|| device.supports_dtype(FloatDType::F16)).unwrap_or(false);
    Some(Found {
        name: identity.name,
        dtype: if half {
            FloatDType::F16
        } else {
            FloatDType::F32
        },
        device,
    })
}

/// The CPU, which always works. It trains in f32: it can compute in f16, but at half the speed.
pub fn cpu() -> Found {
    Found {
        name: "CPU".to_string(),
        device: Device::flex(),
        dtype: FloatDType::F32,
    }
}

/// Fails with advice when a model with `parameters` weights can't train on `found`, so
/// nobody downloads gigabytes to find out.
///
/// wgpu doesn't say how much memory a GPU has, so this briefly holds a quarter more than
/// the weights take. Training needs more than that, so only models with no chance are
/// turned down; the rest can still run out later. The CPU isn't probed: running out of
/// system RAM ends the process instead of panicking.
///
/// `packed` says how many of the weights QLoRA holds in 4 bits: none without it.
pub fn check_memory(found: &Found, parameters: usize, packed: usize) -> Result<()> {
    if found.device == Device::flex() {
        return Ok(());
    }
    // 4 MB each. Burn's pool sizes its pages after the largest allocation it has seen,
    // so bigger chunks cost the training that follows up to 260 MB of GPU memory.
    const CHUNK: usize = 4 << 20;
    let bytes_per_weight = if found.dtype == FloatDType::F16 { 2 } else { 4 };
    // A packed weight takes 4 bits, and each block of them shares a 32-bit scale.
    let packed_bits = 4 * packed + 32 * packed.div_ceil(usize::from(crate::qlora::BLOCK));
    let weights = (parameters - packed) * bytes_per_weight + packed_bits / 8;
    let needed = weights + weights / 4;
    let advice = if packed == 0 {
        "Try --qlora to hold it in 4 bits, a smaller model"
    } else {
        "Try a smaller model"
    };
    let probe = guard(|| {
        let held: Vec<Tensor<1>> = (0..needed.div_ceil(CHUNK))
            .map(|_| Tensor::empty([CHUNK / size_of::<f32>()], &found.device))
            .collect();
        // A failed allocation only shows once something is read back from the GPU.
        if let Some(last) = held.last() {
            last.clone().slice(0..1).into_scalar::<f32>();
        }
    });
    // The chunks are released lazily, and a cleanup that comes before that keeps some.
    found.device.sync().ok();
    found.device.memory_cleanup();
    probe.map_err(|_| {
        anyhow!(
            "{} doesn't have the {:.1} GB of memory this model needs. {advice} \
             such as qwen2.5:0.5b, or --cpu to use system RAM instead",
            found.name,
            needed as f64 / 1e9
        )
    })
}

/// Runs `f`, turning a panic inside it into an error that says what to try.
/// wgpu reports running out of memory and driver faults by panicking, often on a worker
/// thread, which would otherwise end the program with a raw message and a backtrace hint.
/// Guards take turns, because the panic hook they swap is shared by the whole process.
/// So `f` must not call `guard` itself.
pub fn guard<T>(f: impl FnOnce() -> T) -> Result<T> {
    static TURN: Mutex<()> = Mutex::new(());
    let _turn = TURN.lock().unwrap_or_else(|poisoned| poisoned.into_inner());
    // The first panic, often on a worker thread, names the cause; the main thread later
    // fails with a vaguer message. So remember every message, not only the last one.
    let messages = Arc::new(Mutex::new(Vec::new()));
    let seen = Arc::clone(&messages);
    let default_hook = panic::take_hook();
    panic::set_hook(Box::new(move |info| {
        if let Ok(mut seen) = seen.lock() {
            seen.push(info.to_string());
        }
    }));
    let result = panic::catch_unwind(AssertUnwindSafe(f));
    panic::set_hook(default_hook);
    result.map_err(|_| {
        let messages = messages.lock().map(|m| m.clone()).unwrap_or_default();
        if messages
            .iter()
            .any(|m| m.to_lowercase().contains("out of memory"))
        {
            return anyhow!(
                "the device ran out of memory. Try --qlora, --batch-size 1, --max-len 512, \
                 a smaller model, or --cpu to use system RAM instead"
            );
        }
        let first = messages
            .first()
            .and_then(|m| m.lines().nth(1).or(m.lines().next()));
        anyhow!(
            "the compute backend failed: {}. Try --cpu, or update the graphics driver",
            first.unwrap_or("no details")
        )
    })
}

/// Old drivers can panic deep inside wgpu. That should mark a device as broken, not crash us.
fn quietly<T>(f: impl FnOnce() -> T) -> Option<T> {
    guard(f).ok()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn cpu_is_listed_last_and_works() {
        let cpu = list().pop().expect("list is never empty");
        assert_eq!(cpu.name, "CPU");
        assert!(works(&cpu.device));
    }

    #[test]
    fn the_cpu_is_never_probed_for_memory() {
        assert!(check_memory(&cpu(), usize::MAX / 8, 0).is_ok());
    }

    #[test]
    fn out_of_memory_becomes_advice() {
        let error = guard(|| {
            let worker = std::thread::spawn(|| panic!("wgpu error: Out of Memory"));
            let _ = worker.join();
            panic!("Memory location was never initialized");
        })
        .unwrap_err();
        assert!(error.to_string().contains("--batch-size 1"), "{error}");
        let error = guard(|| panic!("lost device")).unwrap_err();
        assert!(error.to_string().contains("lost device"), "{error}");
    }
}
