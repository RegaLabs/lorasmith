//! Finding hardware to train on.

use std::panic::{self, AssertUnwindSafe};

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

fn quietly<T>(f: impl FnOnce() -> T) -> Option<T> {
    // Old drivers can panic deep inside wgpu. That should mark a device as broken, not crash us.
    let default_hook = panic::take_hook();
    panic::set_hook(Box::new(|_| {}));
    let result = panic::catch_unwind(AssertUnwindSafe(f));
    panic::set_hook(default_hook);
    result.ok()
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
}
