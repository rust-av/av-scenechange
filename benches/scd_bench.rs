#![allow(missing_docs)]
#![allow(clippy::unwrap_used)]

use std::{
    fs::File,
    hint::black_box,
    io::{BufReader, Read},
};

#[cfg(feature = "ffmpeg")]
use av_decoders::FfmpegDecoder;
#[cfg(feature = "vapoursynth")]
use av_decoders::VapoursynthDecoder;
use av_decoders::{Decoder, Y4mDecoder};
use av_scenechange::{DetectionOptions, SceneDetectionSpeed, detect_scene_changes};
use criterion::{Criterion, criterion_group, criterion_main};

const TEST_FILE: &str = "./test_files/tt_sif.y4m"; // 112
const LONG_TEST_FILE: &str = "./test_files/deadline_qcif.y4m"; // 1374

const DEFAULT_OPTIONS: DetectionOptions = DetectionOptions {
    analysis_speed: SceneDetectionSpeed::Standard,
    detect_flashes: true,
    min_scenecut_distance: Some(24),
    max_scenecut_distance: Some(250),
    lookahead_distance: 5,
};

fn y4m_benchmark(c: &mut Criterion) {
    c.bench_function("y4m detect", |b| {
        b.iter_batched(
            || {
                let file = black_box(File::open(TEST_FILE).unwrap());
                let reader = black_box(BufReader::new(file));
                let decoder = Decoder::from_decoder_impl(av_decoders::DecoderImpl::Y4m(black_box(
                    Y4mDecoder::new(Box::new(reader) as Box<dyn Read>).unwrap(),
                )))
                .unwrap();
                let bit_depth = decoder.get_video_details().bit_depth;

                (decoder, bit_depth, DEFAULT_OPTIONS)
            },
            |(mut decoder, bit_depth, options)| {
                let _ = match bit_depth {
                    8 => detect_scene_changes::<u8>(&mut decoder, options, None, None).ok(),
                    _ => detect_scene_changes::<u16>(&mut decoder, options, None, None).ok(),
                };
            },
            criterion::BatchSize::LargeInput,
        )
    });
}

fn y4m_long_benchmark(c: &mut Criterion) {
    let mut group = c.benchmark_group("y4m long");
    group.sample_size(20);
    group.bench_function("y4m long detect", |b| {
        b.iter_batched(
            || {
                let file = black_box(File::open(LONG_TEST_FILE).unwrap());
                let reader = black_box(BufReader::new(file));
                let decoder = Decoder::from_decoder_impl(av_decoders::DecoderImpl::Y4m(black_box(
                    Y4mDecoder::new(Box::new(reader) as Box<dyn Read>).unwrap(),
                )))
                .unwrap();
                let bit_depth = decoder.get_video_details().bit_depth;

                (decoder, bit_depth, DEFAULT_OPTIONS)
            },
            |(mut decoder, bit_depth, options)| {
                let _ = match bit_depth {
                    8 => detect_scene_changes::<u8>(&mut decoder, options, None, None).ok(),
                    _ => detect_scene_changes::<u16>(&mut decoder, options, None, None).ok(),
                };
            },
            criterion::BatchSize::LargeInput,
        )
    });
    group.finish();
}

#[cfg(feature = "vapoursynth")]
fn vapoursynth_benchmark(c: &mut Criterion) {
    c.bench_function("vapoursynth detect", |b| {
        use std::collections::HashMap;

        let script = format!(
            r#"
import vapoursynth as vs
core = vs.core
clip = core.lsmas.LWLibavSource(source="{}")
clip.set_output(0)
"#,
            TEST_FILE
        );
        // Create the decoder once to build the index file
        let _ = Decoder::from_decoder_impl(av_decoders::DecoderImpl::Vapoursynth(black_box(
            VapoursynthDecoder::from_script(&script, HashMap::new()).unwrap(),
        )))
        .unwrap();

        b.iter_batched(
            || {
                let decoder = Decoder::from_decoder_impl(av_decoders::DecoderImpl::Vapoursynth(
                    black_box(VapoursynthDecoder::from_script(&script, HashMap::new()).unwrap()),
                ))
                .unwrap();
                let bit_depth = decoder.get_video_details().bit_depth;

                (decoder, bit_depth, DEFAULT_OPTIONS)
            },
            |(mut decoder, bit_depth, options)| {
                let _ = match bit_depth {
                    8 => detect_scene_changes::<u8>(&mut decoder, options, None, None).ok(),
                    _ => detect_scene_changes::<u16>(&mut decoder, options, None, None).ok(),
                };
            },
            criterion::BatchSize::LargeInput,
        )
    });
}

#[cfg(feature = "ffmpeg")]
fn ffmpeg_benchmark(c: &mut Criterion) {
    c.bench_function("ffmpeg decode", |b| {
        b.iter_batched(
            || {
                let decoder = Decoder::from_decoder_impl(av_decoders::DecoderImpl::Ffmpeg(
                    black_box(FfmpegDecoder::new(TEST_FILE).unwrap()),
                ))
                .unwrap();
                let bit_depth = decoder.get_video_details().bit_depth;

                (decoder, bit_depth, DEFAULT_OPTIONS)
            },
            |(mut decoder, bit_depth, options)| {
                let _ = match bit_depth {
                    8 => detect_scene_changes::<u8>(&mut decoder, options, None, None).ok(),
                    _ => detect_scene_changes::<u16>(&mut decoder, options, None, None).ok(),
                };
            },
            criterion::BatchSize::LargeInput,
        )
    });
}

#[cfg(not(feature = "vapoursynth"))]
fn vapoursynth_benchmark(_c: &mut Criterion) {
}

#[cfg(not(feature = "ffmpeg"))]
fn ffmpeg_benchmark(_c: &mut Criterion) {
}

criterion_group!(
    scd_bench,
    y4m_benchmark,
    y4m_long_benchmark,
    vapoursynth_benchmark,
    ffmpeg_benchmark,
    // y4m_hbd_benchmark,
    // vapoursynth_benchmark,
    // vapoursynth_hbd_benchmark,
    // vapoursynth_python_downscale_benchmark,
    // vapoursynth_downscale_benchmark,
    // vapoursynth_empty_benchmark,
    // ffmpeg_benchmark,
    // ffmpeg_hbd_benchmark,
);
criterion_main!(scd_bench);
