#![allow(missing_docs)]
#![allow(clippy::unwrap_used)]

use std::{
    fs::File,
    hint::black_box,
    io::{BufReader, Read},
    sync::Arc,
};

use av_decoders::{Decoder, Y4mDecoder};
use av_scenechange::{
    _bench_internals::{
        Fixed,
        FrameMEStats,
        estimate_importance_block_difference,
        estimate_inter_costs,
        estimate_intra_costs,
    },
    Rational32,
};
use criterion::{Criterion, criterion_group, criterion_main};
use v_frame::{chroma::ChromaSubsampling, frame::Frame, pixel::Pixel};

const TEST_FILE: &str = "./test_files/tt_sif.y4m";
const HBD_TEST_FILE: &str = "./test_files/tt_sif_10bit.y4m";

struct TestData<T: Pixel> {
    frame1: Arc<Frame<T>>,
    frame2: Arc<Frame<T>>,
    width: usize,
    height: usize,
    bit_depth: usize,
    frame_rate: Rational32,
    chroma_sampling: ChromaSubsampling,
}

fn decode_test_data<T: Pixel>(path: &str) -> TestData<T> {
    let file = File::open(path).unwrap();
    let reader = BufReader::new(file);
    let mut decoder = Decoder::from_decoder_impl(av_decoders::DecoderImpl::Y4m(
        Y4mDecoder::new(Box::new(reader) as Box<dyn Read>).unwrap(),
    ))
    .unwrap();

    let details = decoder.get_video_details();
    let width = details.width;
    let height = details.height;
    let bit_depth = details.bit_depth;
    let frame_rate = details.frame_rate.recip();
    let chroma_sampling = details.chroma_sampling;

    let frame1: Arc<Frame<T>> = Arc::new(decoder.read_video_frame().unwrap());
    let frame2: Arc<Frame<T>> = Arc::new(decoder.read_video_frame().unwrap());

    TestData {
        frame1,
        frame2,
        width,
        height,
        bit_depth,
        frame_rate,
        chroma_sampling,
    }
}

fn bench_intra_costs_8bit(c: &mut Criterion) {
    let data = decode_test_data::<u8>(TEST_FILE);
    c.bench_function("estimate_intra_costs 8-bit", |b| {
        b.iter_batched(
            || data.frame2.y_plane.clone(),
            |mut temp_plane| {
                estimate_intra_costs(
                    black_box(&mut temp_plane),
                    black_box(&data.frame2),
                    black_box(data.bit_depth),
                );
            },
            criterion::BatchSize::LargeInput,
        );
    });
}

fn bench_intra_costs_10bit(c: &mut Criterion) {
    let data = decode_test_data::<u16>(HBD_TEST_FILE);
    c.bench_function("estimate_intra_costs 10-bit", |b| {
        b.iter_batched(
            || data.frame2.y_plane.clone(),
            |mut temp_plane| {
                estimate_intra_costs(
                    black_box(&mut temp_plane),
                    black_box(&data.frame2),
                    black_box(data.bit_depth),
                );
            },
            criterion::BatchSize::LargeInput,
        );
    });
}

fn bench_inter_costs_8bit(c: &mut Criterion) {
    let data = decode_test_data::<u8>(TEST_FILE);
    let cols = 2 * data.width.align_power_of_two_and_shift(3);
    let rows = 2 * data.height.align_power_of_two_and_shift(3);
    c.bench_function("estimate_inter_costs 8-bit", |b| {
        b.iter_batched(
            || FrameMEStats::new_arc_array(cols, rows),
            |buffer| {
                estimate_inter_costs(
                    black_box(&data.frame2),
                    black_box(&data.frame1),
                    black_box(data.bit_depth),
                    black_box(data.frame_rate),
                    black_box(data.chroma_sampling),
                    buffer,
                );
            },
            criterion::BatchSize::LargeInput,
        );
    });
}

fn bench_inter_costs_10bit(c: &mut Criterion) {
    let data = decode_test_data::<u16>(HBD_TEST_FILE);
    let cols = 2 * data.width.align_power_of_two_and_shift(3);
    let rows = 2 * data.height.align_power_of_two_and_shift(3);
    c.bench_function("estimate_inter_costs 10-bit", |b| {
        b.iter_batched(
            || FrameMEStats::new_arc_array(cols, rows),
            |buffer| {
                estimate_inter_costs(
                    black_box(&data.frame2),
                    black_box(&data.frame1),
                    black_box(data.bit_depth),
                    black_box(data.frame_rate),
                    black_box(data.chroma_sampling),
                    buffer,
                );
            },
            criterion::BatchSize::LargeInput,
        );
    });
}

fn bench_importance_block_diff_8bit(c: &mut Criterion) {
    let data = decode_test_data::<u8>(TEST_FILE);
    c.bench_function("estimate_importance_block_difference 8-bit", |b| {
        b.iter(|| {
            estimate_importance_block_difference(black_box(&data.frame2), black_box(&data.frame1));
        });
    });
}

fn bench_importance_block_diff_10bit(c: &mut Criterion) {
    let data = decode_test_data::<u16>(HBD_TEST_FILE);
    c.bench_function("estimate_importance_block_difference 10-bit", |b| {
        b.iter(|| {
            estimate_importance_block_difference(black_box(&data.frame2), black_box(&data.frame1));
        });
    });
}

criterion_group!(
    micro_benches,
    bench_intra_costs_8bit,
    bench_intra_costs_10bit,
    bench_inter_costs_8bit,
    bench_inter_costs_10bit,
    bench_importance_block_diff_8bit,
    bench_importance_block_diff_10bit,
);
criterion_main!(micro_benches);
