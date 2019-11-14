use av_scenechange::*;
use clap::{App, Arg};
use std::fs::File;
use std::io::BufReader;

fn main() {
    let matches = App::new("av-scenechange")
        .arg(
            Arg::with_name("INPUT")
                .help("Sets the input file to use")
                .required(true)
                .index(1),
        )
        .get_matches();
    let filename = matches.value_of("INPUT").unwrap();
    let opts = DetectionOptions::default();
    let file = File::open(filename).expect("Failed to read input file");
    let mut reader = BufReader::new(file);
    let mut dec = y4m::Decoder::new(&mut reader).unwrap();
    let bit_depth = dec.get_bit_depth();
    let results = if bit_depth == 8 {
        detect_scene_changes::<_, u8>(&mut dec, opts)
    } else {
        detect_scene_changes::<_, u16>(&mut dec, opts)
    };
    print!("{}", serde_json::to_string(&results).unwrap());
}
