#![deny(clippy::all)]
#![warn(clippy::nursery)]
#![warn(clippy::pedantic)]
#![allow(clippy::cast_possible_truncation)]
#![allow(clippy::cast_possible_wrap)]
#![allow(clippy::cast_precision_loss)]
#![allow(clippy::cast_sign_loss)]
#![allow(clippy::default_trait_access)]
#![allow(clippy::inconsistent_struct_constructor)]
#![allow(clippy::inline_always)]
#![allow(clippy::module_name_repetitions)]
#![allow(clippy::redundant_closure_for_method_calls)]
#![allow(clippy::similar_names)]
#![allow(clippy::struct_excessive_bools)]
#![allow(clippy::use_self)]
#![warn(clippy::clone_on_ref_ptr)]
#![warn(clippy::create_dir)]
#![warn(clippy::dbg_macro)]
#![warn(clippy::default_numeric_fallback)]
#![warn(clippy::exit)]
#![warn(clippy::filetype_is_file)]
#![warn(clippy::float_cmp_const)]
#![warn(clippy::if_then_some_else_none)]
#![warn(clippy::lossy_float_literal)]
#![warn(clippy::map_err_ignore)]
#![warn(clippy::mem_forget)]
#![warn(clippy::mod_module_files)]
#![warn(clippy::multiple_inherent_impl)]
#![warn(clippy::pattern_type_mismatch)]
#![warn(clippy::rc_buffer)]
#![warn(clippy::rc_mutex)]
#![warn(clippy::rest_pat_in_fully_bound_structs)]
#![warn(clippy::same_name_method)]
#![warn(clippy::str_to_string)]
#![warn(clippy::string_to_string)]
#![warn(clippy::undocumented_unsafe_blocks)]
#![warn(clippy::unnecessary_self_imports)]
#![warn(clippy::unneeded_field_pattern)]
#![warn(clippy::use_debug)]
#![warn(clippy::verbose_file_reads)]
// For binary-only crates
#![allow(clippy::missing_errors_doc)]
#![allow(clippy::missing_panics_doc)]

use std::{
    fs::File,
    io::{self, BufReader, Read, Write},
};

use av_scenechange::{detect_scene_changes, DetectionOptions, SceneDetectionSpeed};
use clap::Parser;

#[derive(Parser, Debug)]
struct Args {
    /// Sets the input file to use
    #[clap(value_parser)]
    pub input: String,

    /// Optional file to write results to
    #[clap(long, short, value_parser)]
    pub output: Option<String>,

    /// Speed level for scene-change detection, 0: best quality, 1: fastest mode
    #[clap(long, short, value_parser, default_value_t = 0)]
    pub speed: u8,

    /// Do not detect short scene flashes and exclude them as scene cuts
    #[clap(long)]
    pub no_flash_detection: bool,

    /// Sets a minimum interval between two consecutive scenecuts
    #[clap(long, value_parser)]
    pub min_scenecut: Option<usize>,

    /// Sets a maximum interval between two consecutive scenecuts,
    /// after which a scenecut will be forced
    #[clap(long, value_parser)]
    pub max_scenecut: Option<usize>,
}

fn main() {
    #[cfg(feature = "tracing")]
    use rust_hawktracer::*;
    init_logger();

    #[cfg(feature = "tracing")]
    let instance = HawktracerInstance::new();
    #[cfg(feature = "tracing")]
    let _listener = instance.create_listener(HawktracerListenerType::ToFile {
        file_path: "trace.bin".into(),
        buffer_size: 4096,
    });

    let matches = Args::parse();
    let input = match matches.input.as_str() {
        "-" => Box::new(io::stdin()) as Box<dyn Read>,
        f => Box::new(File::open(f).unwrap()) as Box<dyn Read>,
    };
    let mut reader = BufReader::new(input);

    let mut opts = DetectionOptions {
        detect_flashes: !matches.no_flash_detection,
        min_scenecut_distance: matches.min_scenecut,
        max_scenecut_distance: matches.max_scenecut,
        ..DetectionOptions::default()
    };

    opts.analysis_speed = match matches.speed {
        0 => SceneDetectionSpeed::Standard,
        1 => SceneDetectionSpeed::Fast,
        _ => panic!("Speed mode must be in range [0; 1]"),
    };

    let mut dec = y4m::Decoder::new(&mut reader).unwrap();
    let bit_depth = dec.get_bit_depth();
    let results = if bit_depth == 8 {
        detect_scene_changes::<_, u8>(&mut dec, opts, None, None)
    } else {
        detect_scene_changes::<_, u16>(&mut dec, opts, None, None)
    };
    print!("{}", serde_json::to_string(&results).unwrap());

    if let Some(output_file) = matches.output {
        let mut file = File::create(output_file).expect("Could not create file");

        let output =
            serde_json::to_string_pretty(&results).expect("Could not convert results into json");
        file.write_all(&output.into_bytes()).unwrap();
    }
}

#[cfg(not(feature = "devel"))]
const fn init_logger() {
    // Do nothing
}

#[cfg(feature = "devel")]
fn init_logger() {
    use std::str::FromStr;
    fn level_colored(l: log::Level) -> console::StyledObject<&'static str> {
        use console::style;
        use log::Level;
        match l {
            Level::Trace => style("??").dim(),
            Level::Debug => style("? ").dim(),
            Level::Info => style("> ").green(),
            Level::Warn => style("! ").yellow(),
            Level::Error => style("!!").red(),
        }
    }

    let level = std::env::var("RAV1E_LOG")
        .ok()
        .and_then(|l| log::LevelFilter::from_str(&l).ok())
        .unwrap_or(log::LevelFilter::Info);

    fern::Dispatch::new()
        .format(move |out, message, record| {
            out.finish(format_args!(
                "{level} {message}",
                level = level_colored(record.level()),
                message = message,
            ));
        })
        // set the default log level. to filter out verbose log messages from dependencies, set
        // this to Warn and overwrite the log level for your crate.
        .level(log::LevelFilter::Warn)
        // change log levels for individual modules. Note: This looks for the record's target
        // field which defaults to the module path but can be overwritten with the `target`
        // parameter:
        // `info!(target="special_target", "This log message is about special_target");`
        .level_for("rav1e", level)
        // output to stdout
        .chain(std::io::stderr())
        .apply()
        .unwrap();
}
