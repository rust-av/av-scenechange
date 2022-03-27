use std::{
    fs::File,
    io::{self, BufReader, Read, Write},
};

use av_scenechange::{detect_scene_changes, DetectionOptions, SceneDetectionSpeed};
use clap::{Arg, Command};

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

    let matches = Command::new("av-scenechange")
        .arg(
            Arg::new("INPUT")
                .help("Sets the input file to use")
                .required(true)
                .index(1),
        )
        .arg(
            Arg::new("SPEED_MODE")
                .help("Speed level for scene-change detection, 0: best quality, 1: fastest mode")
                .long("speed")
                .short('s')
                .takes_value(true)
                .default_value("0"),
        )
        .arg(
            Arg::new("NO_FLASH_DETECT")
                .help("Do not detect short scene flashes and exclude them as scene cuts")
                .long("no-flash-detection"),
        )
        .arg(
            Arg::new("MIN_KEYINT")
                .help("Sets a minimum interval between two consecutive scenecuts")
                .long("min-scenecut")
                .takes_value(true),
        )
        .arg(
            Arg::new("MAX_KEYINT")
                .help(
                    "Sets a maximum interval between two consecutive scenecuts, after which a \
                     scenecut will be forced",
                )
                .long("max-scenecut")
                .takes_value(true),
        )
        .arg(
            Arg::new("OUTPUT")
                .help("File to write results in")
                .long("output")
                .short('o')
                .takes_value(true),
        )
        .get_matches();
    let input = match matches.value_of("INPUT").unwrap() {
        "-" => Box::new(io::stdin()) as Box<dyn Read>,
        f => Box::new(File::open(&f).unwrap()) as Box<dyn Read>,
    };
    let mut reader = BufReader::new(input);

    let mut opts = DetectionOptions {
        detect_flashes: !matches.is_present("NO_FLASH_DETECT"),
        min_scenecut_distance: matches.value_of("MIN_KEYINT").map(|val| {
            val.parse()
                .expect("Min-scenecut must be a positive integer")
        }),
        max_scenecut_distance: matches.value_of("MAX_KEYINT").map(|val| {
            val.parse()
                .expect("Max-scenecut must be a positive integer")
        }),
        ..DetectionOptions::default()
    };

    if let Some(speed_mode) = matches.value_of("SPEED_MODE") {
        opts.analysis_speed = match speed_mode {
            "0" => SceneDetectionSpeed::Standard,
            "1" => SceneDetectionSpeed::Fast,
            _ => panic!("Speed mode must be in range [0; 1]"),
        };
    }

    let mut dec = y4m::Decoder::new(&mut reader).unwrap();
    let bit_depth = dec.get_bit_depth();
    let results = if bit_depth == 8 {
        detect_scene_changes::<_, u8>(&mut dec, opts, None, None)
    } else {
        detect_scene_changes::<_, u16>(&mut dec, opts, None, None)
    };
    print!("{}", serde_json::to_string(&results).unwrap());

    if matches.is_present("OUTPUT") {
        let output_file = matches.value_of("OUTPUT").unwrap();
        let mut file = File::create(&output_file).expect("Could not create file");

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
