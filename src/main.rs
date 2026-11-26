use std::io::{IsTerminal, Write};
use std::path::{Path, PathBuf};
use std::time::{Duration, Instant};

use anyhow::{Context, Result};
use burn::tensor::FloatDType;
use clap::{Parser, Subcommand};
use lorasmith::config::Config;
use lorasmith::train::{Options, Progress};
use lorasmith::{data, device, export, hub, load, qlora, schedule, train};

#[derive(Parser)]
#[command(version, about)]
struct Cli {
    #[command(subcommand)]
    command: Command,
}

#[derive(Subcommand)]
enum Command {
    /// Fine-tune a model on a JSONL file and write an adapter, a merged model and a Modelfile.
    Train(TrainArgs),
    /// Show the devices lorasmith can train on and which one it will pick.
    Devices,
}

#[derive(clap::Args)]
struct TrainArgs {
    /// A short name such as qwen2.5:0.5b, a Hugging Face id, or a local model folder.
    model: String,
    /// JSONL with `messages`, `prompt`/`completion` or `text` on each line.
    data: PathBuf,
    /// Where to write the results. Defaults to runs/<data file name>.
    #[arg(long)]
    out: Option<PathBuf>,
    /// Passes over the data. Defaults to about 75 steps' worth, between 3 and 10.
    #[arg(long)]
    epochs: Option<usize>,
    #[arg(long, default_value_t = Options::default().learning_rate)]
    learning_rate: f64,
    /// Examples per step. At least 1.
    #[arg(long, default_value_t = Options::default().batch_size, value_parser = at_least(1))]
    batch_size: usize,
    /// LoRA rank. Higher learns more and uses more memory. At least 1.
    #[arg(long, default_value_t = Options::default().rank, value_parser = at_least(1))]
    rank: usize,
    /// Longer examples are cut to this many tokens. At least 2, since each token learns
    /// to predict the next one.
    #[arg(long, default_value_t = 1024, value_parser = at_least(2))]
    max_len: usize,
    /// Train on the CPU even when a GPU works.
    #[arg(long)]
    cpu: bool,
    /// Hold the model in 4 bits while training. Needs far less memory, at a small cost in
    /// exactness; the merged model is still built on the full weights.
    #[arg(long)]
    qlora: bool,
    /// Run each layer again during the backward pass instead of keeping its activations.
    /// Far less memory for long examples and the same results, for a second forward pass.
    #[arg(long)]
    checkpoint: bool,
    /// Stretch the context of a model pretrained without RoPE scaling. Only `yarn`, as
    /// Qwen documents for long inputs; the merged model and GGUF file keep it.
    #[arg(long, value_parser = ["yarn"])]
    rope_scaling: Option<String>,
    /// How many times longer the context gets with --rope-scaling. Above 1.
    #[arg(long, default_value_t = 4.0, requires = "rope_scaling", value_parser = above_one)]
    rope_factor: f64,
}

fn above_one(text: &str) -> Result<f64, String> {
    match text.parse::<f64>() {
        Ok(n) if n > 1.0 && n.is_finite() => Ok(n),
        _ => Err("use a number above 1, such as 4".to_string()),
    }
}

/// Parses a whole number no smaller than `min`. Clap puts the flag name in front of the error.
fn at_least(min: usize) -> impl Fn(&str) -> Result<usize, String> + Clone + Send + Sync {
    move |text| match text.parse::<usize>() {
        Ok(n) if n >= min => Ok(n),
        _ => Err(format!("use a whole number of {min} or more")),
    }
}

fn main() -> Result<()> {
    match Cli::parse().command {
        Command::Train(args) => train(args),
        Command::Devices => {
            devices();
            Ok(())
        }
    }
}

fn train(args: TrainArgs) -> Result<()> {
    let terminal = std::io::stdout().is_terminal();
    let found = if args.cpu {
        device::cpu()
    } else {
        device::best()
    };
    let text = data::read(&args.data)?;
    let shape = |config: &Config| -> Result<Config> {
        let mut config = config.clone();
        if args.rope_scaling.is_some() {
            config.stretch(args.rope_factor)?;
        }
        Ok(config)
    };
    let model = hub::fetch(&args.model, terminal, |config| {
        let config = shape(config)?;
        config.check_max_len(args.max_len)?;
        let size = config.parameters();
        println!(
            "Model   {} ({:.2}B parameters)",
            args.model,
            size as f64 / 1e9
        );
        let packed = if args.qlora {
            config.projection_parameters()
        } else {
            0
        };
        device::check_memory(&found, size, packed)
    })?;
    let config = shape(&model.config)?;
    if let Some(yarn) = config.yarn().filter(|_| args.rope_scaling.is_some()) {
        println!(
            "Context {} tokens, stretched {}x with YaRN",
            config.context(),
            yarn.factor
        );
    }
    let dataset = data::load(&text, &args.data, &model.dir, args.max_len)?;
    let (examples, validation) = schedule::split(dataset.examples, Options::default().seed);
    println!(
        "Data    {} examples ({} cut to {} tokens, {} skipped, {} held out to check progress)",
        examples.len() + validation.len(),
        dataset.truncated,
        args.max_len,
        dataset.skipped,
        validation.len()
    );

    let half = found.dtype == FloatDType::F16;
    println!(
        "Device  {}{}{}",
        found.name,
        if half { ", in half precision" } else { "" },
        if args.qlora {
            ", over a 4-bit base"
        } else {
            ""
        }
    );
    let device = found.device.autodiff();
    let options = Options {
        rank: args.rank,
        alpha: 2.0 * args.rank as f64,
        learning_rate: args.learning_rate,
        epochs: args
            .epochs
            .unwrap_or_else(|| schedule::epochs_for(examples.len(), args.batch_size)),
        batch_size: args.batch_size,
        checkpoint: args.checkpoint,
        ..Options::default()
    };
    let steps = examples.len().div_ceil(options.batch_size) * options.epochs;
    println!("Train   {} epochs, {steps} steps", options.epochs);
    let out = args.out.unwrap_or_else(|| free_run_dir(&args.data));
    // If training stops early, the adapter from the last finished epoch is still here.
    let checkpoint = out.join("checkpoint");
    std::fs::create_dir_all(&checkpoint)
        .with_context(|| format!("can't create {}", checkpoint.display()))?;
    let start = Instant::now();
    let mut after_first = None;
    let mut scores = Vec::new();
    let trained = device::guard(|| -> Result<_> {
        let base = if args.qlora {
            qlora::load(&config, &model.dir, &device, found.dtype)?
        } else {
            load::model_as(&config, &model.dir, &device, found.dtype)?
        };
        let base = train::attach(base, &options);
        train::train(base, &examples, &validation, &options, &device, |p| {
            // The first step compiles GPU kernels, so it says little about the rest.
            let first = *after_first.get_or_insert_with(Instant::now);
            report(&p, first.elapsed(), terminal);
            scores.extend(p.validation);
            if p.epoch_done
                && let Err(error) = export::adapter(p.model, &model.id, &checkpoint)
            {
                println!("Note    couldn't save a checkpoint: {error:#}");
            }
        })
    })
    .and_then(|trained| trained)
    // A progress line may still be open on the terminal.
    .inspect_err(|_| println!())?;

    let best = (0..scores.len()).min_by(|&a, &b| scores[a].total_cmp(&scores[b]));
    if let Some(best) = best.filter(|&best| best + 1 < scores.len()) {
        println!(
            "Kept    epoch {}, which did best on the held-out examples",
            best + 1
        );
    }

    let skipped = device::guard(|| export::all(trained, &model.id, &model.dir, &config, &out))??;
    // The finished run holds the final adapter; the checkpoint was only a safety net.
    std::fs::remove_dir_all(&checkpoint).ok();
    if let Some(reason) = skipped {
        println!("\nNote    no GGUF file: {reason:#}");
        println!(
            "        Ollama can still run merged/ on Apple Silicon and NVIDIA RTX 20 series or newer."
        );
    }
    if let Some(yarn) = config.yarn().filter(|_| args.rope_scaling.is_some()) {
        println!(
            "\nNote    the adapter alone doesn't carry --rope-scaling. Load its base model with\n        \
             rope_scaling {{\"rope_type\": \"yarn\", \"factor\": {}, \
             \"original_max_position_embeddings\": {}}}, as merged/config.json has it.",
            yarn.factor, yarn.original_context
        );
    }
    let name = out
        .file_name()
        .and_then(|n| n.to_str())
        .unwrap_or("my-model");
    println!(
        "\nSaved   {} in {}",
        out.display(),
        minutes(start.elapsed())
    );
    println!(
        "\nChat with it:\n  ollama create {name} -f {}\n  ollama run {name}",
        out.join("Modelfile").display()
    );
    Ok(())
}

/// `since_first` is the time spent on every step after the first.
fn report(progress: &Progress, since_first: Duration, terminal: bool) {
    let left = match progress.step {
        1 => "estimating".to_string(),
        step => {
            let per_step = since_first / (step - 1) as u32;
            format!(
                "{} left",
                minutes(per_step * (progress.total - step) as u32)
            )
        }
    };
    let line = format!(
        "Step    {}/{}  loss {:.3}  {left}",
        progress.step, progress.total, progress.loss
    );
    if terminal {
        print!("\r{line}   ");
        std::io::stdout().flush().ok();
        if progress.step == progress.total {
            println!();
        }
    } else if progress.step.is_multiple_of((progress.total / 10).max(1))
        || progress.step == progress.total
    {
        println!("{line}");
    }
    if let Some(score) = progress.validation {
        if terminal && progress.step < progress.total {
            println!();
        }
        println!("Epoch   {}  validation loss {score:.3}", progress.epoch);
    }
}

/// runs/<data name>, or runs/<data name>-2 and so on, so earlier results are never overwritten.
fn free_run_dir(data: &Path) -> PathBuf {
    let stem = data.file_stem().and_then(|s| s.to_str()).unwrap_or("run");
    (1..)
        .map(|n| match n {
            1 => PathBuf::from("runs").join(stem),
            n => PathBuf::from("runs").join(format!("{stem}-{n}")),
        })
        .find(|dir| !dir.exists())
        .expect("some numbered folder is free")
}

fn minutes(duration: Duration) -> String {
    let seconds = duration.as_secs();
    format!("{}m{:02}s", seconds / 60, seconds % 60)
}

fn devices() {
    for found in device::list() {
        let status = if device::works(&found.device) {
            "ok"
        } else {
            "not working"
        };
        println!("{:<44} {status}", found.name);
    }
    println!("\nTraining runs on: {}", device::best().name);
    println!(
        "Pick another GPU with CUBECL_WGPU_DEFAULT_DEVICE=IntegratedGpu(0) or DiscreteGpu(1)."
    );
}

#[cfg(test)]
mod tests {
    use super::*;

    fn parse(extra: &[&str]) -> Result<Cli, clap::Error> {
        let args = ["lorasmith", "train", "qwen2.5:0.5b", "data.jsonl"];
        Cli::try_parse_from(args.iter().chain(extra))
    }

    #[test]
    fn rejects_sizes_that_cannot_train() {
        assert!(parse(&["--batch-size", "0"]).is_err());
        assert!(parse(&["--max-len", "1"]).is_err());
        assert!(parse(&["--rank", "0"]).is_err());
        assert!(parse(&["--batch-size", "1", "--max-len", "2", "--rank", "1"]).is_ok());
    }

    #[test]
    fn rope_factor_needs_rope_scaling_and_a_real_stretch() {
        assert!(parse(&["--rope-factor", "4"]).is_err());
        assert!(parse(&["--rope-scaling", "linear"]).is_err());
        assert!(parse(&["--rope-scaling", "yarn", "--rope-factor", "1"]).is_err());
        assert!(parse(&["--rope-scaling", "yarn", "--rope-factor", "inf"]).is_err());
        let Command::Train(args) = parse(&["--rope-scaling", "yarn"]).unwrap().command else {
            panic!("expected train");
        };
        assert_eq!(args.rope_factor, 4.0);
    }
}
