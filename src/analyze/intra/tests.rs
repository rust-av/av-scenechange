use std::{
    mem::MaybeUninit,
    num::{NonZeroU8, NonZeroUsize},
};

use aligned::Aligned;
use cfg_if::cfg_if;
use v_frame::{chroma::ChromaSubsampling, frame::FrameBuilder, pixel::Pixel, plane::Plane};

use super::{IntraEdge, IntraEdgeBuffer, MAX_TX_SIZE};
use crate::data::{
    block::TxSize,
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

    let width_nz = NonZeroUsize::new(width).expect("width must be non-zero");
    let height_nz = NonZeroUsize::new(height).expect("height must be non-zero");

    // Determine bit depth based on pixel type
    // For u8, use 8-bit; for u16, use 10-bit as a reasonable default
    let bit_depth = if std::mem::size_of::<T>() == 1 {
        NonZeroU8::new(8).expect("8 is non-zero")
    } else {
        NonZeroU8::new(10).expect("10 is non-zero")
    };

    // Calculate padding needed to achieve the desired stride
    let padding_right = stride - width;

    // Create a monochrome frame and extract the y_plane
    let frame = FrameBuilder::new(
        width_nz,
        height_nz,
        ChromaSubsampling::Monochrome,
        bit_depth,
    )
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

#[test]
fn predict_dc_intra_variant_none_u8() {
    let bit_depth = 8;
    let tx_size = TxSize::TX_4X4;
    let width = tx_size.width();
    let height = tx_size.height();

    let mut plane = create_test_plane::<u8>(width, height, width);
    let mut dst = plane.region_mut(Area::Rect(Rect {
        x: 0,
        y: 0,
        width,
        height,
    }));

    // For NONE variant, no edge pixels are used
    let mut edge_buf = Aligned([MaybeUninit::uninit(); 4 * MAX_TX_SIZE + 1]);
    let edge = create_test_edge_buf(&mut edge_buf, &[], &[], 0u8);

    predict_dc_intra_internal_verify_asm(
        PredictionVariant::NONE,
        &mut dst,
        tx_size,
        bit_depth,
        &edge,
    );

    // Expected value for 8-bit depth is 128
    let expected_value = 128u8;
    for y in 0..height {
        for x in 0..width {
            assert_eq!(dst[y][x], expected_value, "Mismatch at ({}, {})", x, y);
        }
    }
}

#[test]
fn predict_dc_intra_variant_none_u16() {
    let bit_depth = 10;
    let tx_size = TxSize::TX_4X4;
    let width = tx_size.width();
    let height = tx_size.height();

    let mut plane = create_test_plane::<u16>(width, height, width);
    let mut dst = plane.region_mut(Area::Rect(Rect {
        x: 0,
        y: 0,
        width,
        height,
    }));

    let mut edge_buf = Aligned([MaybeUninit::uninit(); 4 * MAX_TX_SIZE + 1]);
    let edge = create_test_edge_buf(&mut edge_buf, &[], &[], 0u16);

    predict_dc_intra_internal_verify_asm(
        PredictionVariant::NONE,
        &mut dst,
        tx_size,
        bit_depth,
        &edge,
    );

    // Expected value for 10-bit depth is 128 << (10-8) = 512
    let expected_value = 512u16;
    for y in 0..height {
        for x in 0..width {
            assert_eq!(dst[y][x], expected_value, "Mismatch at ({}, {})", x, y);
        }
    }
}

#[test]
fn predict_dc_intra_variant_left() {
    let bit_depth = 8;
    let tx_size = TxSize::TX_4X4;
    let width = tx_size.width();
    let height = tx_size.height();

    let mut plane = create_test_plane::<u8>(width, height, width);
    let mut dst = plane.region_mut(Area::Rect(Rect {
        x: 0,
        y: 0,
        width,
        height,
    }));

    // Left pixels: [200, 100, 150, 50] (bottom to top)
    let left_pixels = [200u8, 100, 150, 50];
    let mut edge_buf = Aligned([MaybeUninit::uninit(); 4 * MAX_TX_SIZE + 1]);
    let edge = create_test_edge_buf(&mut edge_buf, &left_pixels, &[], 0u8);

    predict_dc_intra_internal_verify_asm(
        PredictionVariant::LEFT,
        &mut dst,
        tx_size,
        bit_depth,
        &edge,
    );

    // Expected average: (200 + 100 + 150 + 50 + 2) / 4 = 125
    let expected_value = 125u8;
    for y in 0..height {
        for x in 0..width {
            assert_eq!(dst[y][x], expected_value, "Mismatch at ({}, {})", x, y);
        }
    }
}

#[test]
fn predict_dc_intra_variant_top() {
    let bit_depth = 8;
    let tx_size = TxSize::TX_4X4;
    let width = tx_size.width();
    let height = tx_size.height();

    let mut plane = create_test_plane::<u8>(width, height, width);
    let mut dst = plane.region_mut(Area::Rect(Rect {
        x: 0,
        y: 0,
        width,
        height,
    }));

    // Top pixels: [80, 120, 160, 240]
    let top_pixels = [80u8, 120, 160, 240];
    let mut edge_buf = Aligned([MaybeUninit::uninit(); 4 * MAX_TX_SIZE + 1]);
    let edge = create_test_edge_buf(&mut edge_buf, &[], &top_pixels, 0u8);

    predict_dc_intra_internal_verify_asm(
        PredictionVariant::TOP,
        &mut dst,
        tx_size,
        bit_depth,
        &edge,
    );

    // Expected average: (80 + 120 + 160 + 240 + 2) / 4 = 150
    let expected_value = 150u8;
    for y in 0..height {
        for x in 0..width {
            assert_eq!(dst[y][x], expected_value, "Mismatch at ({}, {})", x, y);
        }
    }
}

#[test]
fn predict_dc_intra_variant_both() {
    let bit_depth = 8;
    let tx_size = TxSize::TX_4X4;
    let width = tx_size.width();
    let height = tx_size.height();

    let mut plane = create_test_plane::<u8>(width, height, width);
    let mut dst = plane.region_mut(Area::Rect(Rect {
        x: 0,
        y: 0,
        width,
        height,
    }));

    // Left pixels: [100, 140, 180, 220] (bottom to top)
    // Top pixels: [110, 130, 170, 190]
    let left_pixels = [100u8, 140, 180, 220];
    let top_pixels = [110u8, 130, 170, 190];
    let mut edge_buf = Aligned([MaybeUninit::uninit(); 4 * MAX_TX_SIZE + 1]);
    let edge = create_test_edge_buf(&mut edge_buf, &left_pixels, &top_pixels, 0u8);

    predict_dc_intra_internal_verify_asm(
        PredictionVariant::BOTH,
        &mut dst,
        tx_size,
        bit_depth,
        &edge,
    );

    // Expected average: (100+140+180+220 + 110+130+170+190 + 4) / 8 = 155
    let expected_value = 155u8;
    for y in 0..height {
        for x in 0..width {
            assert_eq!(dst[y][x], expected_value, "Mismatch at ({}, {})", x, y);
        }
    }
}

#[test]
fn predict_dc_intra_variant_left_u16() {
    let bit_depth = 16;
    let tx_size = TxSize::TX_4X4;
    let width = tx_size.width();
    let height = tx_size.height();

    let mut plane = create_test_plane::<u16>(width, height, width);
    let mut dst = plane.region_mut(Area::Rect(Rect {
        x: 0,
        y: 0,
        width,
        height,
    }));

    // Left pixels: [200, 100, 150, 50] (bottom to top)
    let left_pixels = [200u16, 100, 150, 50];
    let mut edge_buf = Aligned([MaybeUninit::uninit(); 4 * MAX_TX_SIZE + 1]);
    let edge = create_test_edge_buf(&mut edge_buf, &left_pixels, &[], 0u16);

    predict_dc_intra_internal_verify_asm(
        PredictionVariant::LEFT,
        &mut dst,
        tx_size,
        bit_depth,
        &edge,
    );

    // Expected average: (200 + 100 + 150 + 50 + 2) / 4 = 125
    let expected_value = 125u16;
    for y in 0..height {
        for x in 0..width {
            assert_eq!(dst[y][x], expected_value, "Mismatch at ({}, {})", x, y);
        }
    }
}

#[test]
fn predict_dc_intra_variant_top_u16() {
    let bit_depth = 16;
    let tx_size = TxSize::TX_4X4;
    let width = tx_size.width();
    let height = tx_size.height();

    let mut plane = create_test_plane::<u16>(width, height, width);
    let mut dst = plane.region_mut(Area::Rect(Rect {
        x: 0,
        y: 0,
        width,
        height,
    }));

    // Top pixels: [80, 120, 160, 240]
    let top_pixels = [80u16, 120, 160, 240];
    let mut edge_buf = Aligned([MaybeUninit::uninit(); 4 * MAX_TX_SIZE + 1]);
    let edge = create_test_edge_buf(&mut edge_buf, &[], &top_pixels, 0u16);

    predict_dc_intra_internal_verify_asm(
        PredictionVariant::TOP,
        &mut dst,
        tx_size,
        bit_depth,
        &edge,
    );

    // Expected average: (80 + 120 + 160 + 240 + 2) / 4 = 150
    let expected_value = 150u16;
    for y in 0..height {
        for x in 0..width {
            assert_eq!(dst[y][x], expected_value, "Mismatch at ({}, {})", x, y);
        }
    }
}

#[test]
fn predict_dc_intra_variant_both_u16() {
    let bit_depth = 16;
    let tx_size = TxSize::TX_4X4;
    let width = tx_size.width();
    let height = tx_size.height();

    let mut plane = create_test_plane::<u16>(width, height, width);
    let mut dst = plane.region_mut(Area::Rect(Rect {
        x: 0,
        y: 0,
        width,
        height,
    }));

    // Left pixels: [100, 140, 180, 220] (bottom to top)
    // Top pixels: [110, 130, 170, 190]
    let left_pixels = [100u16, 140, 180, 220];
    let top_pixels = [110u16, 130, 170, 190];
    let mut edge_buf = Aligned([MaybeUninit::uninit(); 4 * MAX_TX_SIZE + 1]);
    let edge = create_test_edge_buf(&mut edge_buf, &left_pixels, &top_pixels, 0u16);

    predict_dc_intra_internal_verify_asm(
        PredictionVariant::BOTH,
        &mut dst,
        tx_size,
        bit_depth,
        &edge,
    );

    // Expected average: (100+140+180+220 + 110+130+170+190 + 4) / 8 = 155
    let expected_value = 155u16;
    for y in 0..height {
        for x in 0..width {
            assert_eq!(dst[y][x], expected_value, "Mismatch at ({}, {})", x, y);
        }
    }
}

#[test]
fn predict_dc_intra_different_sizes() {
    let bit_depth = 8;

    // Test various block sizes
    let test_sizes = [
        TxSize::TX_4X4,
        TxSize::TX_8X8,
        TxSize::TX_16X16,
        TxSize::TX_8X4,
        TxSize::TX_4X8,
    ];

    for tx_size in test_sizes {
        let width = tx_size.width();
        let height = tx_size.height();

        let mut plane = create_test_plane::<u8>(width, height, width);
        let mut dst = plane.region_mut(Area::Rect(Rect {
            x: 0,
            y: 0,
            width,
            height,
        }));

        // Create uniform edge pixels for predictable result
        let left_pixels = vec![100u8; height];
        let top_pixels = vec![200u8; width];
        let mut edge_buf = Aligned([MaybeUninit::uninit(); 4 * MAX_TX_SIZE + 1]);
        let edge = create_test_edge_buf(&mut edge_buf, &left_pixels, &top_pixels, 0u8);

        predict_dc_intra_internal_verify_asm(
            PredictionVariant::BOTH,
            &mut dst,
            tx_size,
            bit_depth,
            &edge,
        );

        // Expected average: (height*100 + width*200 + (width+height)/2) /
        // (width+height)
        let sum = (height * 100 + width * 200) as u32;
        let len = (width + height) as u32;
        let expected_value = ((sum + (len >> 1)) / len) as u8;

        for y in 0..height {
            for x in 0..width {
                assert_eq!(
                    dst[y][x], expected_value,
                    "Mismatch at ({}, {}) for size {}x{}",
                    x, y, width, height
                );
            }
        }
    }
}

#[test]
fn predict_dc_intra_edge_cases() {
    let bit_depth = 8;
    let tx_size = TxSize::TX_4X4;
    let width = tx_size.width();
    let height = tx_size.height();

    // Test with extreme pixel values
    let test_cases = [
        // (left_pixels, top_pixels, expected_avg_approx)
        (vec![0u8; 4], vec![0u8; 4], 0u8),
        (vec![255u8; 4], vec![255u8; 4], 255u8),
        (vec![0u8; 4], vec![255u8; 4], 127u8),
        (vec![255u8; 4], vec![0u8; 4], 127u8),
    ];

    for (left_pixels, top_pixels, _expected_approx) in test_cases {
        let mut plane = create_test_plane::<u8>(width, height, width);
        let mut dst = plane.region_mut(Area::Rect(Rect {
            x: 0,
            y: 0,
            width,
            height,
        }));

        let mut edge_buf = Aligned([MaybeUninit::uninit(); 4 * MAX_TX_SIZE + 1]);
        let edge = create_test_edge_buf(&mut edge_buf, &left_pixels, &top_pixels, 0u8);

        predict_dc_intra_internal_verify_asm(
            PredictionVariant::BOTH,
            &mut dst,
            tx_size,
            bit_depth,
            &edge,
        );

        // Verify all pixels have the same value (DC prediction should be uniform)
        let first_pixel = dst[0][0];
        for y in 0..height {
            for x in 0..width {
                assert_eq!(
                    dst[y][x], first_pixel,
                    "DC prediction should be uniform, but found different values at ({}, {})",
                    x, y
                );
            }
        }
    }
}

#[test]
fn predict_dc_intra_rounding() {
    let bit_depth = 8;
    let tx_size = TxSize::TX_4X4;
    let width = tx_size.width();
    let height = tx_size.height();

    let mut plane = create_test_plane::<u8>(width, height, width);
    let mut dst = plane.region_mut(Area::Rect(Rect {
        x: 0,
        y: 0,
        width,
        height,
    }));

    // Test case that requires rounding: sum = 7, len = 8
    // Should round to 1 (7 + 4) / 8 = 1
    let left_pixels = [1u8, 1, 1, 1];
    let top_pixels = [1u8, 1, 1, 0];
    let mut edge_buf = Aligned([MaybeUninit::uninit(); 4 * MAX_TX_SIZE + 1]);
    let edge = create_test_edge_buf(&mut edge_buf, &left_pixels, &top_pixels, 0u8);

    predict_dc_intra_internal_verify_asm(
        PredictionVariant::BOTH,
        &mut dst,
        tx_size,
        bit_depth,
        &edge,
    );

    // Sum = 7, len = 8, (7 + 4) / 8 = 1
    let expected_value = 1u8;
    for y in 0..height {
        for x in 0..width {
            assert_eq!(dst[y][x], expected_value, "Mismatch at ({}, {})", x, y);
        }
    }
}

#[test]
fn predict_dc_intra_larger_blocks() {
    let bit_depth = 8;
    let tx_size = TxSize::TX_16X16;
    let width = tx_size.width();
    let height = tx_size.height();

    let mut plane = create_test_plane::<u8>(width, height, width);
    let mut dst = plane.region_mut(Area::Rect(Rect {
        x: 0,
        y: 0,
        width,
        height,
    }));

    // Create predictable gradient patterns
    let left_pixels: Vec<u8> = (0..height).map(|i| (i * 16) as u8).collect();
    let top_pixels: Vec<u8> = (0..width).map(|i| (i * 8) as u8).collect();
    let mut edge_buf = Aligned([MaybeUninit::uninit(); 4 * MAX_TX_SIZE + 1]);
    let edge = create_test_edge_buf(&mut edge_buf, &left_pixels, &top_pixels, 0u8);

    predict_dc_intra_internal_verify_asm(
        PredictionVariant::BOTH,
        &mut dst,
        tx_size,
        bit_depth,
        &edge,
    );

    // Verify uniformity
    let first_pixel = dst[0][0];
    for y in 0..height {
        for x in 0..width {
            assert_eq!(
                dst[y][x], first_pixel,
                "Large block DC prediction should be uniform at ({}, {})",
                x, y
            );
        }
    }
}

#[test]
fn predict_dc_intra_rectangular_blocks() {
    let bit_depth = 8;

    // Test rectangular blocks
    let rectangular_sizes = [
        TxSize::TX_8X4,
        TxSize::TX_4X8,
        TxSize::TX_16X8,
        TxSize::TX_8X16,
    ];

    for tx_size in rectangular_sizes {
        let width = tx_size.width();
        let height = tx_size.height();

        let mut plane = create_test_plane::<u8>(width, height, width);
        let mut dst = plane.region_mut(Area::Rect(Rect {
            x: 0,
            y: 0,
            width,
            height,
        }));

        // Use different values for left and top to ensure proper averaging
        let left_pixels = vec![50u8; height];
        let top_pixels = vec![150u8; width];
        let mut edge_buf = Aligned([MaybeUninit::uninit(); 4 * MAX_TX_SIZE + 1]);
        let edge = create_test_edge_buf(&mut edge_buf, &left_pixels, &top_pixels, 0u8);

        predict_dc_intra_internal_verify_asm(
            PredictionVariant::BOTH,
            &mut dst,
            tx_size,
            bit_depth,
            &edge,
        );

        // Verify all pixels are uniform
        let first_pixel = dst[0][0];
        for y in 0..height {
            for x in 0..width {
                assert_eq!(
                    dst[y][x], first_pixel,
                    "Rectangular block ({} x {}) DC prediction should be uniform at ({}, {})",
                    width, height, x, y
                );
            }
        }

        // The expected value should be weighted by the number of left vs top pixels
        let left_sum = height * 50;
        let top_sum = width * 150;
        let total_sum = (left_sum + top_sum) as u32;
        let total_len = (width + height) as u32;
        let expected = ((total_sum + (total_len >> 1)) / total_len) as u8;

        assert_eq!(
            first_pixel, expected,
            "Incorrect average for {}x{} block",
            width, height
        );
    }
}
