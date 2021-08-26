use av_scenechange::*;
use clap::{App, Arg};
use std::fs::File;
use std::io::{self, BufReader, Read};

fn main() {
    let matches = App::new("av-scenechange")
        .arg(
            Arg::with_name("INPUT")
                .help("Sets the input file to use")
                .required(true)
                .index(1),
        )
        .arg(
            Arg::with_name("FAST_MODE")
                .help("Uses faster but less accurate analysis")
                .long("fast"),
        )
        .arg(
            Arg::with_name("EXCLUDE_FLASHES")
                .help("Detect short scene flashes and exclude them as scene cuts")
                .long("no-flashes"),
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
        .get_matches();
    let input = match matches.value_of("INPUT").unwrap() {
        "-" => Box::new(io::stdin()) as Box<dyn Read>,
        f => Box::new(File::open(&f).unwrap()) as Box<dyn Read>,
    };
    let mut reader = BufReader::new(input);
    let opts = DetectionOptions {
        fast_analysis: matches.is_present("FAST_MODE"),
        ignore_flashes: matches.is_present("EXCLUDE_FLASHES"),
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
    let mut dec = y4m::Decoder::new(&mut reader).unwrap();
    let bit_depth = dec.get_bit_depth();
    let results = if bit_depth == 8 {
        detect_scene_changes::<_, u8>(&mut dec, opts, None)
    } else {
        detect_scene_changes::<_, u16>(&mut dec, opts, None)
    };
    print!("{}", serde_json::to_string(&results).unwrap());
}
