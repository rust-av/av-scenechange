use av_scenechange::*;
use clap::{App, Arg};
use std::fs::File;
use std::io::{self, BufReader, Read, Write};

fn main() {
    let matches = App::new("av-scenechange")
        .arg(
            Arg::with_name("INPUT")
                .help("Sets the input file to use")
                .required(true)
                .index(1),
        )
        .arg(
            Arg::with_name("SPEED_MODE")
                .help("Speed level for scene-change detection, 0: best quality, 1: speed-to-quality trade-off, 2: fastest mode")
                .long("speed")
                .short("s")
                .takes_value(true)
                .default_value("0"),
        )
        .arg(
            Arg::with_name("NO_FLASH_DETECT")
                .help("Do not detect short scene flashes and exclude them as scene cuts")
                .long("no-flash-detection"),
        )
        .arg(
            Arg::with_name("MIN_KEYINT")
                .help("Sets a minimum interval between two consecutive scenecuts")
                .long("min-scenecut")
                .takes_value(true),
        )
        .arg(
            Arg::with_name("MAX_KEYINT")
                .help("Sets a maximum interval between two consecutive scenecuts, after which a scenecut will be forced")
                .long("max-scenecut")
                .takes_value(true),
        )
        .arg(
            Arg::with_name("OUTPUT")
                .help("File to write results in")
                .long("output")
                .short("o")
                .takes_value(true),
        )
        .get_matches();
    let input = match matches.value_of("INPUT").unwrap() {
        "-" => Box::new(io::stdin()) as Box<dyn Read>,
        f => Box::new(File::open(&f).unwrap()) as Box<dyn Read>,
    };
    let mut reader = BufReader::new(input);

    let mut opts = DetectionOptions {
        ignore_flashes: matches.is_present("NO_FLASH_DETECT"),
        min_scenecut_distance: matches.value_of("MIN_KEYINT").map(|val| {
            val.parse()
                .expect("Min-scenecut must be a positive integer")
        }),
        max_scenecut_distance: matches.value_of("MAX_KEYINT").map(|val| {
            val.parse()
                .expect("Max-scenecut must be a positive integer")
        }),
        ..Default::default()
    };

    if let Some(speed_mode) = matches.value_of("SPEED_MODE") {
        opts.fast_analysis = match speed_mode {
            "0" => SceneDetectionSpeed::Slow,
            "1" => SceneDetectionSpeed::Medium,
            "2" => SceneDetectionSpeed::Fast,
            _ => panic!("Speed mode must be in range [0; 2]"),
        };
    }

    let mut dec = y4m::Decoder::new(&mut reader).unwrap();
    let bit_depth = dec.get_bit_depth();
    let results = if bit_depth == 8 {
        detect_scene_changes::<_, u8>(&mut dec, opts, None)
    } else {
        detect_scene_changes::<_, u16>(&mut dec, opts, None)
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
