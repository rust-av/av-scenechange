#![allow(clippy::unwrap_used, reason = "allow in test files")]

use std::num::{NonZeroU8, NonZeroUsize};

use v_frame::{chroma::ChromaSubsampling, frame::FrameBuilder, pixel::Pixel, plane::Plane};

use super::*;

fn create_plane<T: Pixel>(width: usize, height: usize) -> Plane<T> {
    let bit_depth = if size_of::<T>() == 1 { 8 } else { 10 };
    FrameBuilder::new(
        NonZeroUsize::new(width).unwrap(),
        NonZeroUsize::new(height).unwrap(),
        ChromaSubsampling::Monochrome,
        NonZeroU8::new(bit_depth).unwrap(),
    )
    .build::<T>()
    .unwrap()
    .y_plane
}

fn fill_plane<T: Pixel>(plane: &mut Plane<T>, value: i32) {
    let stride = plane.geometry().stride.get();
    let width = plane.width().get();
    let height = plane.height().get();
    let origin = plane.data_origin();
    let data = plane.data_mut();
    for row in 0..height {
        for col in 0..width {
            data[origin + row * stride + col] = T::from(value).unwrap();
        }
    }
}

fn fill_plane_gradient<T: Pixel>(plane: &mut Plane<T>) {
    let stride = plane.geometry().stride.get();
    let width = plane.width().get();
    let height = plane.height().get();
    let origin = plane.data_origin();
    let data = plane.data_mut();
    for row in 0..height {
        for col in 0..width {
            let val = ((row * width + col) % 256) as i32;
            data[origin + row * stride + col] = T::from(val).unwrap();
        }
    }
}

fn assert_sse2_matches_rust<T: Pixel>(plane: &Plane<T>, x: usize, y: usize) {
    let rust_result = sum_8x8_block_rust(plane, x, y);
    let dispatch_result = sum_8x8_block(plane, x, y);
    assert_eq!(
        rust_result, dispatch_result,
        "SSE2 mismatch at block ({x}, {y}): rust={rust_result}, sse2={dispatch_result}"
    );
}

#[test]
fn sum_8x8_block_u16_zeros() {
    let mut plane = create_plane::<u16>(16, 16);
    fill_plane(&mut plane, 0);
    assert_sse2_matches_rust(&plane, 0, 0);
    assert_eq!(sum_8x8_block(&plane, 0, 0), 0);
}

#[test]
fn sum_8x8_block_u16_max_12bit() {
    let mut plane = create_plane::<u16>(16, 16);
    fill_plane(&mut plane, 4095);
    assert_sse2_matches_rust(&plane, 0, 0);
    assert_eq!(sum_8x8_block(&plane, 0, 0), 4095 * 64);
}

#[test]
fn sum_8x8_block_u8_zeros() {
    let mut plane = create_plane::<u8>(16, 16);
    fill_plane(&mut plane, 0);
    assert_sse2_matches_rust(&plane, 0, 0);
    assert_eq!(sum_8x8_block(&plane, 0, 0), 0);
}

#[test]
fn sum_8x8_block_u8_max() {
    let mut plane = create_plane::<u8>(16, 16);
    fill_plane(&mut plane, 255);
    assert_sse2_matches_rust(&plane, 0, 0);
    assert_eq!(sum_8x8_block(&plane, 0, 0), 255 * 64);
}

#[test]
fn sum_8x8_block_u16_gradient() {
    let mut plane = create_plane::<u16>(16, 16);
    fill_plane_gradient(&mut plane);
    assert_sse2_matches_rust(&plane, 0, 0);
    assert_sse2_matches_rust(&plane, 1, 0);
    assert_sse2_matches_rust(&plane, 0, 1);
    assert_sse2_matches_rust(&plane, 1, 1);
}

#[test]
fn sum_8x8_block_u8_gradient() {
    let mut plane = create_plane::<u8>(16, 16);
    fill_plane_gradient(&mut plane);
    assert_sse2_matches_rust(&plane, 0, 0);
    assert_sse2_matches_rust(&plane, 1, 0);
    assert_sse2_matches_rust(&plane, 0, 1);
    assert_sse2_matches_rust(&plane, 1, 1);
}

#[test]
fn sum_8x8_block_u16_multiple_positions() {
    let mut plane = create_plane::<u16>(64, 64);
    fill_plane_gradient(&mut plane);
    for y in 0..8 {
        for x in 0..8 {
            assert_sse2_matches_rust(&plane, x, y);
        }
    }
}

#[test]
fn sum_8x8_block_u8_multiple_positions() {
    let mut plane = create_plane::<u8>(64, 64);
    fill_plane_gradient(&mut plane);
    for y in 0..8 {
        for x in 0..8 {
            assert_sse2_matches_rust(&plane, x, y);
        }
    }
}
