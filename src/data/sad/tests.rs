use std::num::{NonZeroU8, NonZeroUsize};

use cfg_if::cfg_if;
use v_frame::{chroma::ChromaSubsampling, frame::FrameBuilder, pixel::Pixel, plane::Plane};

use crate::data::{
    plane::{Area, AsRegion},
    sad::get_sad,
};

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

/// Helper function to create a simple test plane without padding
fn create_test_plane<T: Pixel>(width: usize, height: usize) -> Plane<T> {
    create_padded_plane(width, height, 0)
}

// Generate plane data for get_sad tests
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

// Generate plane data for sad_plane_internal tests
fn setup_equal_stride_planes<T: Pixel>() -> (Plane<T>, Plane<T>) {
    // Two planes with same stride for sad_plane_internal testing
    let mut input_plane = create_test_plane::<T>(320, 240);
    let mut rec_plane = create_test_plane::<T>(320, 240);

    let input_geom = input_plane.geometry();
    let rec_geom = rec_plane.geometry();

    for (i, row) in input_plane
        .data_mut()
        .chunks_mut(input_geom.stride.get())
        .enumerate()
    {
        for (j, pixel) in row.iter_mut().enumerate() {
            let val = ((j + i) as i32) & 255i32;
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
            let val = (j as i32 - i as i32) & 255i32;
            assert!(val >= u8::MIN.into() && val <= u8::MAX.into());
            *pixel = T::from(val).expect("value should fit in Pixel");
        }
    }

    (input_plane, rec_plane)
}

fn sad_plane_verify_asm<T: Pixel>(src: &Plane<T>, dst: &Plane<T>) -> u64 {
    let rust_output = super::rust::sad_plane_internal(src, dst);

    cfg_if! {
        if #[cfg(asm_x86_64)] {
            if crate::cpu::has_avx2() {
                let asm_output = unsafe { super::avx2::sad_plane_internal(src, dst) };
                assert_eq!(rust_output, asm_output);
            }
            // All x86_64 CPUs have SSE2
            let asm_output = unsafe { super::sse2::sad_plane_internal(src, dst) };
            assert_eq!(rust_output, asm_output);
        }
    }

    rust_output
}

fn get_sad_same_inner<T: Pixel>() {
    let blocks: Vec<(usize, usize, u32)> = vec![
        (4, 4, 1912),
        (4, 8, 4296),
        (8, 4, 3496),
        (8, 8, 7824),
        (8, 16, 16592),
        (16, 8, 14416),
        (16, 16, 31136),
        (16, 32, 60064),
        (32, 16, 59552),
        (32, 32, 120128),
        (32, 64, 186688),
        (64, 32, 250176),
        (64, 64, 438912),
        (64, 128, 654272),
        (128, 64, 1016768),
        (128, 128, 1689792),
        (4, 16, 8680),
        (16, 4, 6664),
        (8, 32, 31056),
        (32, 8, 27600),
        (16, 64, 93344),
        (64, 16, 116384),
    ];

    let bit_depth: usize = 8;
    let (input_plane, rec_plane) = setup_planes::<T>();

    for (w, h, distortion) in blocks {
        let area = Area::StartingAt { x: 32, y: 40 };

        let input_region = input_plane.region(area);
        let rec_region = rec_plane.region(area);

        assert_eq!(
            distortion,
            get_sad(&input_region, &rec_region, w, h, bit_depth)
        );
    }
}

fn sad_plane_same_inner<T: Pixel>() {
    let (input_plane, rec_plane) = setup_equal_stride_planes::<T>();

    // Test with the full planes
    let _sad_value = sad_plane_verify_asm(&input_plane, &rec_plane);

    // Test with smaller planes to ensure the algorithm works correctly
    let mut small_input = create_test_plane::<T>(16, 16);
    let mut small_rec = create_test_plane::<T>(16, 16);

    let small_input_geom = small_input.geometry();
    let small_rec_geom = small_rec.geometry();

    // Initialize the small planes with test data
    for (i, row) in small_input
        .data_mut()
        .chunks_mut(small_input_geom.stride.get())
        .enumerate()
    {
        for (j, pixel) in row.iter_mut().enumerate() {
            let val = ((j + i) as i32) & 255i32;
            assert!(val >= u8::MIN.into() && val <= u8::MAX.into());
            *pixel = T::from(val).expect("value should fit in Pixel");
        }
    }

    for (i, row) in small_rec
        .data_mut()
        .chunks_mut(small_rec_geom.stride.get())
        .enumerate()
    {
        for (j, pixel) in row.iter_mut().enumerate() {
            let val = (j as i32 - i as i32) & 255i32;
            assert!(val >= u8::MIN.into() && val <= u8::MAX.into());
            *pixel = T::from(val).expect("value should fit in Pixel");
        }
    }

    let _small_sad = sad_plane_verify_asm(&small_input, &small_rec);
}

#[test]
fn get_sad_same_u8() {
    get_sad_same_inner::<u8>();
}

#[test]
fn get_sad_same_u16() {
    get_sad_same_inner::<u16>();
}

#[test]
fn sad_plane_same_u8() {
    sad_plane_same_inner::<u8>();
}

#[test]
fn sad_plane_same_u16() {
    sad_plane_same_inner::<u16>();
}
