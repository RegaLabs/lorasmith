//! Trains from the same adapter as tools/make_lora_fixture.py did with Hugging Face PEFT,
//! and checks that every step's loss and the final adapter match.

use std::collections::HashMap;
use std::path::Path;

use burn::tensor::{Device, FloatDType, Tensor, TensorData, Tolerance};
use loraforge::data::Example;
use loraforge::{load, lora, train};
use serde_json::Value;

const FIXTURE: &str = "tests/fixtures/lora";

/// PEFT tensors as Burn stores them: keyed by Burn path, transposed to [in, r] and [r, out].
fn peft_adapter(dir: &str) -> HashMap<String, TensorData> {
    let bytes = std::fs::read(Path::new(dir).join("adapter_model.safetensors")).unwrap();
    let file = safetensors::SafeTensors::deserialize(&bytes).unwrap();
    let mut factors = HashMap::new();
    for (name, view) in file.tensors() {
        let module = name.strip_prefix("base_model.model.model.").unwrap();
        let (module, factor) = match module.strip_suffix(".lora_A.weight") {
            Some(module) => (module, "a"),
            None => (module.strip_suffix(".lora_B.weight").unwrap(), "b"),
        };
        let values: Vec<f32> = view
            .data()
            .chunks(4)
            .map(|b| f32::from_le_bytes(b.try_into().unwrap()))
            .collect();
        let shape = [view.shape()[0], view.shape()[1]];
        let transposed =
            Tensor::<2>::from_data(TensorData::new(values, shape), &Device::flex()).transpose();
        factors.insert(
            format!("{module}.weight.lora.{factor}"),
            transposed.into_data(),
        );
    }
    factors
}

#[test]
fn training_matches_peft_step_for_step() {
    let expected: Value =
        serde_json::from_str(&std::fs::read_to_string(format!("{FIXTURE}/expected.json")).unwrap())
            .unwrap();
    let examples: Vec<Example> =
        serde_json::from_value::<Vec<HashMap<String, Value>>>(expected["examples"].clone())
            .unwrap()
            .into_iter()
            .map(|e| Example {
                tokens: serde_json::from_value(e["tokens"].clone()).unwrap(),
                train: serde_json::from_value(e["train"].clone()).unwrap(),
            })
            .collect();
    let number = |key: &str| expected[key].as_f64().unwrap();

    let device = Device::flex().autodiff();
    let (_, model) =
        load::model(Path::new("tests/fixtures/llama"), &device, FloatDType::F32).unwrap();
    let model = lora::attach(
        model,
        expected["rank"].as_u64().unwrap() as usize,
        number("alpha"),
    );
    let start = peft_adapter(&format!("{FIXTURE}/start"));
    let mut model = lora::set_factors(model, |path, _| start.get(path).cloned());

    let mut optimizer = train::optimizer();
    let batch: Vec<&Example> = examples.iter().collect();
    for (i, want) in expected["losses"].as_array().unwrap().iter().enumerate() {
        let loss;
        (model, loss) = train::step(
            model,
            &mut optimizer,
            &batch,
            number("learning_rate"),
            number("max_grad_norm"),
            &mut 1.0,
            &device,
        );
        let want = want.as_f64().unwrap() as f32;
        assert!(
            (loss - want).abs() < 1e-5,
            "step {}: loss {loss}, PEFT {want}",
            i + 1
        );
    }

    let end = peft_adapter(&format!("{FIXTURE}/end"));
    let mut compared = 0;
    for (name, linear) in lora::projections(&model) {
        let adapter = linear.weight.adapter().unwrap();
        let path = format!("{}.weight.lora", name.strip_prefix("model.").unwrap());
        for (factor, value) in [("a", adapter.a.val()), ("b", adapter.b.val())] {
            let want = &end[&format!("{path}.{factor}")];
            value
                .into_data()
                .assert_approx_eq::<f32>(want, Tolerance::absolute(1e-5));
            compared += 1;
        }
    }
    assert_eq!(compared, end.len());
}
