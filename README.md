# av-scenechange

[![Actions Status](https://github.com/rust-av/av-scenechange/workflows/av-scenechange/badge.svg)](https://github.com/rust-av/av-scenechange/actions)
[![docs.rs](https://img.shields.io/docsrs/av-scenechange)](https://docs.rs/av-scenechange/latest/av-scenechange/)
[![Crates.io Version](https://img.shields.io/crates/v/av-scenechange)](https://crates.io/crates/av-scenechange)
[![Crates.io License](https://img.shields.io/crates/l/av-scenechange)](LICENSE)

Scenechange detection tool based on rav1e's scene detection code. It is focused around detecting scenechange points that will be optimal for an encoder to place keyframes. It may not be the best tool if your use case is to generate scene changes as a human would interpret them--for that there are other tools such as SCXvid and WWXD.

## Usage

### Command Line

The basic usage of `av-scenechange` is:

```bash
av-scenechange input.y4m
```

This will output the scenechange detection results as JSON to stdout.

#### Options

- `-o, --output <FILE>`: Write results to a file instead of stdout
- `-s, --speed <LEVEL>`: Set detection speed (0 = best quality, 1 = fastest mode, default: 0)
- `--no-flash-detection`: Disable detection of short scene flashes
- `--min-scenecut <FRAMES>`: Set minimum interval between consecutive scenecuts
- `--max-scenecut <FRAMES>`: Set maximum interval between consecutive scenecuts (forces a scenecut)

#### Examples

```bash
# Basic usage with Y4M input
av-scenechange video.y4m

# Save results to a file
av-scenechange input.y4m -o results.json

# Use faster but less accurate detection mode
av-scenechange input.y4m --speed 1

# Set minimum distance between scenecuts to 24 frames
av-scenechange input.y4m --min-scenecut 24

# Read from stdin
cat input.y4m | av-scenechange -
```

### Library Usage

`av-scenechange` can also be used as a Rust library:

```rust
use av_scenechange::{detect_scene_changes, DetectionOptions, SceneDetectionSpeed};

let mut decoder = // ... initialize your decoder
let options = DetectionOptions {
    analysis_speed: SceneDetectionSpeed::Standard,
    detect_flashes: true,
    min_scenecut_distance: Some(24),
    max_scenecut_distance: Some(250),
    ..DetectionOptions::default()
};

let results = detect_scene_changes(&mut decoder, options, None, None)?;
```

## Compiling

### Prerequisites

- **Rust**: Minimum version 1.86
- **NASM**: Required for optimized assembly code (can be disabled with `--no-default-features`)

#### Installing NASM

**Ubuntu/Debian:**

```bash
sudo apt install nasm
```

**Fedora/RHEL:**

```bash
sudo dnf install nasm
```

**macOS:**

```bash
brew install nasm
```

**Windows:**
Download from [nasm.us](https://www.nasm.us/) or use package managers like chocolatey or scoop.

### Building from Source

```bash
# Clone the repository
git clone https://github.com/rust-av/av-scenechange.git
cd av-scenechange

# Build with default features (includes optimized assembly)
cargo build --release

# Build without assembly optimizations (no NASM required)
cargo build --release --no-default-features --features binary,parallel

# Build without parallel processing (single-threaded)
cargo build --release --no-default-features --features binary,asm

# Build with additional features
cargo build --release --features ffmpeg  # FFmpeg input support
cargo build --release --features vapoursynth  # VapourSynth input support
```

### Install from Crates.io

```bash
# Install the latest version (automatically uses release optimizations)
cargo install av-scenechange

# Install with ffmpeg and vapoursynth input support
cargo install av-scenechange --features ffmpeg,vapoursynth
```

This will install the binary to `~/.cargo/bin/av-scenechange`.

### Available Features

- `binary` (default): Enables command-line interface
- `asm` (default): Enables optimized assembly code (requires NASM)
- `parallel` (default): Enables parallel processing using Rayon for improved performance
- `serialize`: Enables JSON serialization support
- `ffmpeg`: Adds FFmpeg decoder support
- `vapoursynth`: Adds VapourSynth decoder support
- `devel`: Development features (logging, console output)
- `tracing`: Chrome tracing support for performance profiling
