use clap::{Parser, Subcommand};
use loraforge::device;

#[derive(Parser)]
#[command(version, about)]
struct Cli {
    #[command(subcommand)]
    command: Command,
}

#[derive(Subcommand)]
enum Command {
    /// Show the devices loraforge can train on and which one it will pick.
    Devices,
}

fn main() {
    match Cli::parse().command {
        Command::Devices => devices(),
    }
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
