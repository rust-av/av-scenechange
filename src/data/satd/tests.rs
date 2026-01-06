use std::num::{NonZeroU8, NonZeroUsize};

use cfg_if::cfg_if;
use v_frame::{chroma::ChromaSubsampling, frame::FrameBuilder, pixel::Pixel, plane::Plane};

use crate::data::plane::{Area, AsRegion, PlaneRegion};

/// Helper function to create a test plane with padding
fn create_padded_plane<T: Pixel>(width: usize, height: usize, padding: usize) -> Plane<T> {
    let width_nz = NonZeroUsize::new(width).expect("width must be non-zero");
    let height_nz = NonZeroUsize::new(height).expect("height must be non-zero");

    // Determine bit depth based on pixel type
    let bit_depth = if std::mem::size_of::<T>() == 1 {
        NonZeroU8::new(8).expect("8 is non-zero")
    } else {
        NonZeroU8::new(10).expect("10 is non-zero")
    };

    // Create a monochrome frame with padding and extract the y_plane
    let frame = FrameBuilder::new(
        width_nz,
        height_nz,
        ChromaSubsampling::Monochrome,
        bit_depth,
    )
    .luma_padding_left(padding)
    .luma_padding_right(padding)
    .luma_padding_top(padding)
    .luma_padding_bottom(padding)
    .build::<T>()
    .expect("Failed to build frame");

    frame.y_plane
}

// Generate plane data for get_sad_same()
fn setup_planes<T: Pixel>() -> (Plane<T>, Plane<T>) {
    // Two planes with different padding
    let mut input_plane = create_padded_plane::<T>(640, 480, 128 + 8);
    let mut rec_plane = create_padded_plane::<T>(640, 480, 2 * 128 + 8);

    // Make the test pattern robust to data alignment
    let input_geom = input_plane.geometry();
    let rec_geom = rec_plane.geometry();
    // In the old API, xpad_off was calculated as (xorigin - xpad) - 8
    // With the way padding worked in the old API, this ended up being -8
    let xpad_off = -8i32;

    for (i, row) in input_plane
        .data_mut()
        .chunks_mut(input_geom.stride.get())
        .enumerate()
    {
        for (j, pixel) in row.iter_mut().enumerate() {
            let val = ((j + i) as i32 - xpad_off) & 255i32;
            assert!(val >= u8::MIN.into() && val <= u8::MAX.into());
            *pixel = T::from(val).expect("value should fit in Pixel");
        }
    }

    for (i, row) in rec_plane
        .data_mut()
        .chunks_mut(rec_geom.stride.get())
        .enumerate()
    {
        for (j, pixel) in row.iter_mut().enumerate() {
            let val = (j as i32 - i as i32 - xpad_off) & 255i32;
            assert!(val >= u8::MIN.into() && val <= u8::MAX.into());
            *pixel = T::from(val).expect("value should fit in Pixel");
        }
    }

    (input_plane, rec_plane)
}

fn get_satd_verify_asm<T: Pixel>(
    src: &PlaneRegion<'_, T>,
    dst: &PlaneRegion<'_, T>,
    w: usize,
    h: usize,
    bit_depth: usize,
) -> u32 {
    let rust_output = super::rust::get_satd_internal(src, dst, w, h, bit_depth);

    cfg_if! {
        if #[cfg(asm_x86_64)] {
            if crate::cpu::has_avx2() {
                // SAFETY: call to SIMD function
                let asm_output = unsafe { super::avx2::get_satd_internal(src, dst, w, h, bit_depth) };
                assert_eq!(rust_output, asm_output);
            }
            if crate::cpu::has_sse4() {
                // SAFETY: call to SIMD function
                let asm_output = unsafe { super::sse4::get_satd_internal(src, dst, w, h, bit_depth) };
                assert_eq!(rust_output, asm_output);
            }
            if crate::cpu::has_ssse3() {
                // SAFETY: call to SIMD function
                let asm_output = unsafe { super::ssse3::get_satd_internal(src, dst, w, h, bit_depth) };
                assert_eq!(rust_output, asm_output);
            }
        } else if #[cfg(asm_neon)] {
            // SAFETY: call to SIMD function
            let asm_output = unsafe { super::neon::get_satd_internal(src, dst, w, h, bit_depth) };
            assert_eq!(rust_output, asm_output);
        }
    }

    rust_output
}

fn get_satd_same_inner<T: Pixel>() {
    let blocks: Vec<(usize, usize, u32)> = vec![
        (4, 4, 1408),
        (4, 8, 2016),
        (8, 4, 1816),
        (8, 8, 3984),
        (8, 16, 5136),
        (16, 8, 4864),
        (16, 16, 9984),
        (16, 32, 13824),
        (32, 16, 13760),
        (32, 32, 27952),
        (32, 64, 37168),
        (64, 32, 45104),
        (64, 64, 84176),
        (64, 128, 127920),
        (128, 64, 173680),
        (128, 128, 321456),
        (4, 16, 3136),
        (16, 4, 2632),
        (8, 32, 7056),
        (32, 8, 6624),
        (16, 64, 18432),
        (64, 16, 21312),
    ];

    let bit_depth: usize = 8;
    let (input_plane, rec_plane) = setup_planes::<T>();

    for (w, h, distortion) in blocks {
        let area = Area::StartingAt { x: 32, y: 40 };

        let input_region = input_plane.region(area);
        let rec_region = rec_plane.region(area);

        assert_eq!(
            distortion,
            get_satd_verify_asm(&input_region, &rec_region, w, h, bit_depth)
        );
    }
}

#[test]
fn get_satd_same_u8() {
    get_satd_same_inner::<u8>();
}

#[test]
fn get_satd_same_u16() {
    get_satd_same_inner::<u16>();
}
