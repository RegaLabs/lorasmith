//! Finding hardware to train on.

use std::panic::{self, AssertUnwindSafe};
use std::sync::{Arc, Mutex};

use anyhow::{Result, anyhow};
use burn::tensor::{Device, DeviceKind, DeviceType, Distribution, Tensor};

/// A device with a name a person recognizes.
pub struct Found {
    pub name: String,
    pub device: Device,
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
    let identity = quietly(|| device.identity()).flatten()?;
    // Software rasterizers like llvmpipe pose as GPUs but lose badly to the CPU backend.
    identity.physical.as_ref()?;
    Some(Found {
        name: identity.name,
        device,
    })
}

/// The CPU, which always works.
pub fn cpu() -> Found {
    Found {
        name: "CPU".to_string(),
        device: Device::flex(),
    }
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
                "the device ran out of memory. Try --batch-size 1, --max-len 512, \
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
