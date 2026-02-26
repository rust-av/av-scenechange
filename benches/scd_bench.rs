#![allow(missing_docs)]
#![allow(clippy::unwrap_used)]

use std::{
    fs::File,
    hint::black_box,
    io::{BufReader, Read},
};

use av_decoders::{Decoder, Y4mDecoder};
use av_scenechange::{DetectionOptions, SceneDetectionSpeed, detect_scene_changes};
use criterion::{Criterion, criterion_group, criterion_main};

const TEST_FILE: &str = "./test_files/tt_sif.y4m";
const HBD_TEST_FILE: &str = "./test_files/tt_sif_10bit.y4m";

const DEFAULT_OPTIONS: DetectionOptions = DetectionOptions {
    analysis_speed: SceneDetectionSpeed::Standard,
    detect_flashes: true,
    min_scenecut_distance: Some(24),
    max_scenecut_distance: Some(250),
    lookahead_distance: 5,
};

fn y4m_8bit(c: &mut Criterion) {
    c.bench_function("y4m detect 8-bit", |b| {
        b.iter_batched(
            || {
                let file = black_box(File::open(TEST_FILE).unwrap());
                let reader = black_box(BufReader::new(file));
                let decoder = Decoder::from_decoder_impl(av_decoders::DecoderImpl::Y4m(black_box(
                    Y4mDecoder::new(Box::new(reader) as Box<dyn Read>).unwrap(),
                )))
                .unwrap();

                (decoder, DEFAULT_OPTIONS)
            },
            |(mut decoder, options)| {
                detect_scene_changes::<u8>(&mut decoder, options, None, None).ok();
            },
            criterion::BatchSize::LargeInput,
        );
    });
}

fn y4m_10bit(c: &mut Criterion) {
    c.bench_function("y4m detect 10-bit", |b| {
        b.iter_batched(
            || {
                let file = black_box(File::open(HBD_TEST_FILE).unwrap());
                let reader = black_box(BufReader::new(file));
                let decoder = Decoder::from_decoder_impl(av_decoders::DecoderImpl::Y4m(black_box(
                    Y4mDecoder::new(Box::new(reader) as Box<dyn Read>).unwrap(),
                )))
                .unwrap();

                (decoder, DEFAULT_OPTIONS)
            },
            |(mut decoder, options)| {
                detect_scene_changes::<u16>(&mut decoder, options, None, None).ok()
            },
            criterion::BatchSize::LargeInput,
        );
    });
}

criterion_group!(scd_bench, y4m_8bit, y4m_10bit);
criterion_main!(scd_bench);
