//! Trains a small random model on whatever device lorasmith picks on this machine.
//! Run with `cargo test --release --test gpu -- --ignored --nocapture`.

use std::time::Instant;

use lorasmith::config::Config;
use lorasmith::data::Example;
use lorasmith::model::Llama;
use lorasmith::train::{self, Options};

const CONFIG: &str = r#"{
    "model_type": "llama", "vocab_size": 2048, "hidden_size": 512, "intermediate_size": 1376,
    "num_hidden_layers": 8, "num_attention_heads": 8, "num_key_value_heads": 4,
    "rms_norm_eps": 1e-6, "max_position_embeddings": 512
}"#;

#[test]
#[ignore = "slow; meant for checking real hardware"]
fn trains_on_the_best_device() {
    let found = lorasmith::device::best();
    let device = found.device.autodiff();
    let config = Config::parse(CONFIG).unwrap();
    let model = train::attach(Llama::new(&config, &device), &Options::default());
    let examples: Vec<Example> = (0..16)
        .map(|i| Example {
            tokens: (0..256).map(|t| ((t * 7 + i) % 2048) as u32).collect(),
            train: vec![true; 256],
        })
        .collect();
    let options = Options {
        learning_rate: 1e-3,
        epochs: 2,
        ..Options::default()
    };

    let start = Instant::now();
    let mut last = start;
    let mut losses = Vec::new();
    train::train(model, &examples, &[], &options, &device, |p| {
        println!(
            "step {}/{}  loss {:.3}  {:.2}s",
            p.step,
            p.total,
            p.loss,
            last.elapsed().as_secs_f32()
        );
        last = Instant::now();
        losses.push(p.loss);
    })
    .unwrap();
    println!("{}: {:.1}s", found.name, start.elapsed().as_secs_f32());
    assert!(losses.last() < losses.first());
}
