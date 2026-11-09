use std::io::{IsTerminal, Write};
use std::path::{Path, PathBuf};
use std::time::{Duration, Instant};

use anyhow::Result;
use clap::{Parser, Subcommand};
use loraforge::train::{Options, Progress};
use loraforge::{data, device, export, hub, load, train};

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
    /// Show the devices loraforge can train on and which one it will pick.
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
    #[arg(long, default_value_t = Options::default().epochs)]
    epochs: usize,
    #[arg(long, default_value_t = Options::default().learning_rate)]
    learning_rate: f64,
    #[arg(long, default_value_t = Options::default().batch_size)]
    batch_size: usize,
    /// LoRA rank. Higher learns more and uses more memory.
    #[arg(long, default_value_t = Options::default().rank)]
    rank: usize,
    /// Longer examples are cut to this many tokens.
    #[arg(long, default_value_t = 1024)]
    max_len: usize,
    /// Train on the CPU even when a GPU works.
    #[arg(long)]
    cpu: bool,
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
    let model = hub::fetch(&args.model, terminal, |config| {
        let size = config.parameters() as f64;
        println!("Model   {} ({:.2}B parameters)", args.model, size / 1e9);
    })?;
    let dataset = data::load(&args.data, &model.dir, args.max_len)?;
    println!(
        "Data    {} examples ({} cut to {} tokens, {} skipped)",
        dataset.examples.len(),
        dataset.truncated,
        args.max_len,
        dataset.skipped
    );

    let found = if args.cpu {
        device::cpu()
    } else {
        device::best()
    };
    println!("Device  {}", found.name);
    let device = found.device.autodiff();
    let (_, base) = load::model(&model.dir, &device)?;
    let options = Options {
        rank: args.rank,
        alpha: 2.0 * args.rank as f64,
        learning_rate: args.learning_rate,
        epochs: args.epochs,
        batch_size: args.batch_size,
        ..Options::default()
    };
    let start = Instant::now();
    let mut after_first = None;
    let trained = train::train(
        train::attach(base, &options),
        &dataset.examples,
        &options,
        &device,
        |p| {
            // The first step compiles GPU kernels, so it says little about the rest.
            let first = *after_first.get_or_insert_with(Instant::now);
            report(&p, first.elapsed(), terminal)
        },
    );

    let out = args.out.unwrap_or_else(|| free_run_dir(&args.data));
    if let Some(reason) = export::all(trained, &model.id, &model.dir, &out)? {
        println!("\nNote    no GGUF file: {reason:#}");
        println!(
            "        Ollama can still run merged/ on Apple Silicon and NVIDIA RTX 20 series or newer."
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
