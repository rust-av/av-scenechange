use std::mem::MaybeUninit;

use aligned::Aligned;
use cfg_if::cfg_if;
use v_frame::{chroma::ChromaSubsampling, frame::FrameBuilder, pixel::Pixel, plane::Plane};

use super::{IntraEdge, IntraEdgeBuffer, MAX_TX_SIZE};
use crate::data::{
    block::TxSize,
    pixel_from_u16,
    plane::{Area, AsRegion, PlaneRegionMut, Rect},
    prediction::PredictionVariant,
};

fn predict_dc_intra_internal_verify_asm<T: Pixel + std::fmt::Debug>(
    variant: PredictionVariant,
    dst: &mut PlaneRegionMut<'_, T>,
    tx_size: TxSize,
    bit_depth: usize,
    edge_buf: &IntraEdge<T>,
) {
    super::rust::predict_dc_intra_internal(variant, dst, tx_size, bit_depth, edge_buf);
    let rust_output = dst.rows_iter().flatten().copied().collect::<Vec<_>>();

    cfg_if! {
        if #[cfg(asm_x86_64)] {
            if crate::cpu::has_avx512icl() {
                // SAFETY: call to SIMD function
                unsafe { super::avx512icl::predict_dc_intra_internal(variant, dst, tx_size, bit_depth, edge_buf); }
                let asm_output = dst.rows_iter().flatten().copied().collect::<Vec<_>>();
                assert_eq!(rust_output, asm_output);
            }
            if crate::cpu::has_avx2() {
                // SAFETY: call to SIMD function
                unsafe { super::avx2::predict_dc_intra_internal(variant, dst, tx_size, bit_depth, edge_buf); }
                let asm_output = dst.rows_iter().flatten().copied().collect::<Vec<_>>();
                assert_eq!(rust_output, asm_output);
            }
            if crate::cpu::has_ssse3() {
                // SAFETY: call to SIMD function
                unsafe { super::ssse3::predict_dc_intra_internal(variant, dst, tx_size, bit_depth, edge_buf); }
                let asm_output = dst.rows_iter().flatten().copied().collect::<Vec<_>>();
                assert_eq!(rust_output, asm_output);
            }
        } else if #[cfg(asm_neon)] {
            // Silence unused warning. In the future maybe we will have NEON ASM here.
            let _output = rust_output;
        }
    }
}

/// Helper function to create a test plane for use in unit tests
fn create_test_plane<T: Pixel>(width: usize, height: usize, stride: usize) -> Plane<T> {
    assert!(stride >= width, "stride must be >= width");

    // Determine bit depth based on pixel type
    // For u8, use 8-bit; for u16, use 10-bit as a reasonable default
    let bit_depth = if std::mem::size_of::<T>() == 1 { 8 } else { 10 };

    // Calculate padding needed to achieve the desired stride
    let padding_right = stride - width;

    // Create a monochrome frame and extract the y_plane
    let frame = FrameBuilder::new(width, height, ChromaSubsampling::Monochrome, bit_depth)
        .luma_padding_right(padding_right)
        .build::<T>()
        .expect("Failed to build frame");

    frame.y_plane
}

/// Helper function to create `IntraEdge` from edge pixel values
fn create_test_edge_buf<'a, T: Pixel>(
    edge_buf: &'a mut IntraEdgeBuffer<T>,
    left_pixels: &[T],
    top_pixels: &[T],
    top_left: T,
) -> IntraEdge<'a, T> {
    // Fill left pixels (bottom to top, right-aligned)
    let left_start = 2 * MAX_TX_SIZE - left_pixels.len();
    for (i, &pixel) in left_pixels.iter().enumerate() {
        edge_buf[left_start + i].write(pixel);
    }

    // Fill top-left pixel
    edge_buf[2 * MAX_TX_SIZE].write(top_left);

    // Fill above pixels
    for (i, &pixel) in top_pixels.iter().enumerate() {
        edge_buf[2 * MAX_TX_SIZE + 1 + i].write(pixel);
    }

    IntraEdge::new(edge_buf, left_pixels.len(), top_pixels.len())
}

struct DcIntraCase<T> {
    name: &'static str,
    variant: PredictionVariant,
    tx_size: TxSize,
    bit_depth: usize,
    left_pixels: Vec<T>,
    top_pixels: Vec<T>,
    expected: T,
}

fn assert_dc_intra_case<T: Pixel + std::fmt::Debug>(case: &DcIntraCase<T>) {
    let width = case.tx_size.width();
    let height = case.tx_size.height();

    let mut plane = create_test_plane::<T>(width, height, width);
    let mut dst = plane.region_mut(Area::Rect(Rect {
        x: 0,
        y: 0,
        width,
        height,
    }));

    let mut edge_buf = Aligned([MaybeUninit::uninit(); 4 * MAX_TX_SIZE + 1]);
    let edge = create_test_edge_buf(
        &mut edge_buf,
        &case.left_pixels,
        &case.top_pixels,
        pixel_from_u16(0),
    );

    predict_dc_intra_internal_verify_asm(
        case.variant,
        &mut dst,
        case.tx_size,
        case.bit_depth,
        &edge,
    );

    for y in 0..height {
        for x in 0..width {
            assert_eq!(
                dst[y][x], case.expected,
                "{} mismatch at ({}, {}) for size {}x{}",
                case.name, x, y, width, height
            );
        }
    }
}

#[test]
fn predict_dc_intra_u8_cases() {
    let cases = [
        DcIntraCase {
            name: "none_4x4",
            variant: PredictionVariant::NONE,
            tx_size: TxSize::TX_4X4,
            bit_depth: 8,
            left_pixels: vec![],
            top_pixels: vec![],
            expected: 128u8,
        },
        DcIntraCase {
            name: "left_4x4",
            variant: PredictionVariant::LEFT,
            tx_size: TxSize::TX_4X4,
            bit_depth: 8,
            left_pixels: vec![200, 100, 150, 50],
            top_pixels: vec![],
            expected: 125u8,
        },
        DcIntraCase {
            name: "top_4x4",
            variant: PredictionVariant::TOP,
            tx_size: TxSize::TX_4X4,
            bit_depth: 8,
            left_pixels: vec![],
            top_pixels: vec![80, 120, 160, 240],
            expected: 150u8,
        },
        DcIntraCase {
            name: "both_4x4",
            variant: PredictionVariant::BOTH,
            tx_size: TxSize::TX_4X4,
            bit_depth: 8,
            left_pixels: vec![100, 140, 180, 220],
            top_pixels: vec![110, 130, 170, 190],
            expected: 155u8,
        },
        DcIntraCase {
            name: "both_8x8_weighted",
            variant: PredictionVariant::BOTH,
            tx_size: TxSize::TX_8X8,
            bit_depth: 8,
            left_pixels: vec![100; TxSize::TX_8X8.height()],
            top_pixels: vec![200; TxSize::TX_8X8.width()],
            expected: 150u8,
        },
        DcIntraCase {
            name: "both_16x16_weighted",
            variant: PredictionVariant::BOTH,
            tx_size: TxSize::TX_16X16,
            bit_depth: 8,
            left_pixels: vec![100; TxSize::TX_16X16.height()],
            top_pixels: vec![200; TxSize::TX_16X16.width()],
            expected: 150u8,
        },
        DcIntraCase {
            name: "both_8x4_weighted",
            variant: PredictionVariant::BOTH,
            tx_size: TxSize::TX_8X4,
            bit_depth: 8,
            left_pixels: vec![100; TxSize::TX_8X4.height()],
            top_pixels: vec![200; TxSize::TX_8X4.width()],
            expected: 167u8,
        },
        DcIntraCase {
            name: "both_4x8_weighted",
            variant: PredictionVariant::BOTH,
            tx_size: TxSize::TX_4X8,
            bit_depth: 8,
            left_pixels: vec![100; TxSize::TX_4X8.height()],
            top_pixels: vec![200; TxSize::TX_4X8.width()],
            expected: 133u8,
        },
        DcIntraCase {
            name: "both_min_edges",
            variant: PredictionVariant::BOTH,
            tx_size: TxSize::TX_4X4,
            bit_depth: 8,
            left_pixels: vec![0; 4],
            top_pixels: vec![0; 4],
            expected: 0u8,
        },
        DcIntraCase {
            name: "both_max_edges",
            variant: PredictionVariant::BOTH,
            tx_size: TxSize::TX_4X4,
            bit_depth: 8,
            left_pixels: vec![255; 4],
            top_pixels: vec![255; 4],
            expected: 255u8,
        },
        DcIntraCase {
            name: "both_min_left_max_top",
            variant: PredictionVariant::BOTH,
            tx_size: TxSize::TX_4X4,
            bit_depth: 8,
            left_pixels: vec![0; 4],
            top_pixels: vec![255; 4],
            expected: 128u8,
        },
        DcIntraCase {
            name: "both_max_left_min_top",
            variant: PredictionVariant::BOTH,
            tx_size: TxSize::TX_4X4,
            bit_depth: 8,
            left_pixels: vec![255; 4],
            top_pixels: vec![0; 4],
            expected: 128u8,
        },
        DcIntraCase {
            name: "both_rounding",
            variant: PredictionVariant::BOTH,
            tx_size: TxSize::TX_4X4,
            bit_depth: 8,
            left_pixels: vec![1, 1, 1, 1],
            top_pixels: vec![1, 1, 1, 0],
            expected: 1u8,
        },
        DcIntraCase {
            name: "both_16x16_gradient",
            variant: PredictionVariant::BOTH,
            tx_size: TxSize::TX_16X16,
            bit_depth: 8,
            left_pixels: (0..TxSize::TX_16X16.height())
                .map(|i| (i * 16) as u8)
                .collect(),
            top_pixels: (0..TxSize::TX_16X16.width())
                .map(|i| (i * 8) as u8)
                .collect(),
            expected: 90u8,
        },
        DcIntraCase {
            name: "both_rect_8x4",
            variant: PredictionVariant::BOTH,
            tx_size: TxSize::TX_8X4,
            bit_depth: 8,
            left_pixels: vec![50; TxSize::TX_8X4.height()],
            top_pixels: vec![150; TxSize::TX_8X4.width()],
            expected: 117u8,
        },
        DcIntraCase {
            name: "both_rect_4x8",
            variant: PredictionVariant::BOTH,
            tx_size: TxSize::TX_4X8,
            bit_depth: 8,
            left_pixels: vec![50; TxSize::TX_4X8.height()],
            top_pixels: vec![150; TxSize::TX_4X8.width()],
            expected: 83u8,
        },
        DcIntraCase {
            name: "both_rect_16x8",
            variant: PredictionVariant::BOTH,
            tx_size: TxSize::TX_16X8,
            bit_depth: 8,
            left_pixels: vec![50; TxSize::TX_16X8.height()],
            top_pixels: vec![150; TxSize::TX_16X8.width()],
            expected: 117u8,
        },
        DcIntraCase {
            name: "both_rect_8x16",
            variant: PredictionVariant::BOTH,
            tx_size: TxSize::TX_8X16,
            bit_depth: 8,
            left_pixels: vec![50; TxSize::TX_8X16.height()],
            top_pixels: vec![150; TxSize::TX_8X16.width()],
            expected: 83u8,
        },
    ];

    for case in cases {
        assert_dc_intra_case(&case);
    }
}

#[test]
fn predict_dc_intra_u16_cases() {
    let cases = [
        DcIntraCase {
            name: "none_4x4_10bpc",
            variant: PredictionVariant::NONE,
            tx_size: TxSize::TX_4X4,
            bit_depth: 10,
            left_pixels: vec![],
            top_pixels: vec![],
            expected: 512u16,
        },
        DcIntraCase {
            name: "left_4x4_16bpc",
            variant: PredictionVariant::LEFT,
            tx_size: TxSize::TX_4X4,
            bit_depth: 16,
            left_pixels: vec![200, 100, 150, 50],
            top_pixels: vec![],
            expected: 125u16,
        },
        DcIntraCase {
            name: "top_4x4_16bpc",
            variant: PredictionVariant::TOP,
            tx_size: TxSize::TX_4X4,
            bit_depth: 16,
            left_pixels: vec![],
            top_pixels: vec![80, 120, 160, 240],
            expected: 150u16,
        },
        DcIntraCase {
            name: "both_4x4_16bpc",
            variant: PredictionVariant::BOTH,
            tx_size: TxSize::TX_4X4,
            bit_depth: 16,
            left_pixels: vec![100, 140, 180, 220],
            top_pixels: vec![110, 130, 170, 190],
            expected: 155u16,
        },
    ];

    for case in cases {
        assert_dc_intra_case(&case);
    }
}
