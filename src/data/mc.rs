use v_frame::{pixel::Pixel, plane::PlaneSlice};

#[cfg(not(any(asm_x86_64, asm_neon)))]
use self::rust::*;
#[cfg(asm_neon)]
use self::simd_neon::*;
#[cfg(asm_x86_64)]
use self::simd_x86::*;
use crate::{cpu::CpuFeatureLevel, data::plane::PlaneRegionMut};

#[derive(Copy, Clone, Debug, PartialEq, Eq, PartialOrd)]
#[allow(clippy::upper_case_acronyms)]
#[allow(dead_code)]
pub enum FilterMode {
    REGULAR = 0,
    SMOOTH = 1,
    SHARP = 2,
    BILINEAR = 3,
    SWITCHABLE = 4,
}

pub const SUBPEL_FILTER_SIZE: usize = 8;

const SUBPEL_FILTERS: [[[i32; SUBPEL_FILTER_SIZE]; 16]; 6] = [
    [
        [0, 0, 0, 128, 0, 0, 0, 0],
        [0, 2, -6, 126, 8, -2, 0, 0],
        [0, 2, -10, 122, 18, -4, 0, 0],
        [0, 2, -12, 116, 28, -8, 2, 0],
        [0, 2, -14, 110, 38, -10, 2, 0],
        [0, 2, -14, 102, 48, -12, 2, 0],
        [0, 2, -16, 94, 58, -12, 2, 0],
        [0, 2, -14, 84, 66, -12, 2, 0],
        [0, 2, -14, 76, 76, -14, 2, 0],
        [0, 2, -12, 66, 84, -14, 2, 0],
        [0, 2, -12, 58, 94, -16, 2, 0],
        [0, 2, -12, 48, 102, -14, 2, 0],
        [0, 2, -10, 38, 110, -14, 2, 0],
        [0, 2, -8, 28, 116, -12, 2, 0],
        [0, 0, -4, 18, 122, -10, 2, 0],
        [0, 0, -2, 8, 126, -6, 2, 0],
    ],
    [
        [0, 0, 0, 128, 0, 0, 0, 0],
        [0, 2, 28, 62, 34, 2, 0, 0],
        [0, 0, 26, 62, 36, 4, 0, 0],
        [0, 0, 22, 62, 40, 4, 0, 0],
        [0, 0, 20, 60, 42, 6, 0, 0],
        [0, 0, 18, 58, 44, 8, 0, 0],
        [0, 0, 16, 56, 46, 10, 0, 0],
        [0, -2, 16, 54, 48, 12, 0, 0],
        [0, -2, 14, 52, 52, 14, -2, 0],
        [0, 0, 12, 48, 54, 16, -2, 0],
        [0, 0, 10, 46, 56, 16, 0, 0],
        [0, 0, 8, 44, 58, 18, 0, 0],
        [0, 0, 6, 42, 60, 20, 0, 0],
        [0, 0, 4, 40, 62, 22, 0, 0],
        [0, 0, 4, 36, 62, 26, 0, 0],
        [0, 0, 2, 34, 62, 28, 2, 0],
    ],
    [
        [0, 0, 0, 128, 0, 0, 0, 0],
        [-2, 2, -6, 126, 8, -2, 2, 0],
        [-2, 6, -12, 124, 16, -6, 4, -2],
        [-2, 8, -18, 120, 26, -10, 6, -2],
        [-4, 10, -22, 116, 38, -14, 6, -2],
        [-4, 10, -22, 108, 48, -18, 8, -2],
        [-4, 10, -24, 100, 60, -20, 8, -2],
        [-4, 10, -24, 90, 70, -22, 10, -2],
        [-4, 12, -24, 80, 80, -24, 12, -4],
        [-2, 10, -22, 70, 90, -24, 10, -4],
        [-2, 8, -20, 60, 100, -24, 10, -4],
        [-2, 8, -18, 48, 108, -22, 10, -4],
        [-2, 6, -14, 38, 116, -22, 10, -4],
        [-2, 6, -10, 26, 120, -18, 8, -2],
        [-2, 4, -6, 16, 124, -12, 6, -2],
        [0, 2, -2, 8, 126, -6, 2, -2],
    ],
    [
        [0, 0, 0, 128, 0, 0, 0, 0],
        [0, 0, 0, 120, 8, 0, 0, 0],
        [0, 0, 0, 112, 16, 0, 0, 0],
        [0, 0, 0, 104, 24, 0, 0, 0],
        [0, 0, 0, 96, 32, 0, 0, 0],
        [0, 0, 0, 88, 40, 0, 0, 0],
        [0, 0, 0, 80, 48, 0, 0, 0],
        [0, 0, 0, 72, 56, 0, 0, 0],
        [0, 0, 0, 64, 64, 0, 0, 0],
        [0, 0, 0, 56, 72, 0, 0, 0],
        [0, 0, 0, 48, 80, 0, 0, 0],
        [0, 0, 0, 40, 88, 0, 0, 0],
        [0, 0, 0, 32, 96, 0, 0, 0],
        [0, 0, 0, 24, 104, 0, 0, 0],
        [0, 0, 0, 16, 112, 0, 0, 0],
        [0, 0, 0, 8, 120, 0, 0, 0],
    ],
    [
        [0, 0, 0, 128, 0, 0, 0, 0],
        [0, 0, -4, 126, 8, -2, 0, 0],
        [0, 0, -8, 122, 18, -4, 0, 0],
        [0, 0, -10, 116, 28, -6, 0, 0],
        [0, 0, -12, 110, 38, -8, 0, 0],
        [0, 0, -12, 102, 48, -10, 0, 0],
        [0, 0, -14, 94, 58, -10, 0, 0],
        [0, 0, -12, 84, 66, -10, 0, 0],
        [0, 0, -12, 76, 76, -12, 0, 0],
        [0, 0, -10, 66, 84, -12, 0, 0],
        [0, 0, -10, 58, 94, -14, 0, 0],
        [0, 0, -10, 48, 102, -12, 0, 0],
        [0, 0, -8, 38, 110, -12, 0, 0],
        [0, 0, -6, 28, 116, -10, 0, 0],
        [0, 0, -4, 18, 122, -8, 0, 0],
        [0, 0, -2, 8, 126, -4, 0, 0],
    ],
    [
        [0, 0, 0, 128, 0, 0, 0, 0],
        [0, 0, 30, 62, 34, 2, 0, 0],
        [0, 0, 26, 62, 36, 4, 0, 0],
        [0, 0, 22, 62, 40, 4, 0, 0],
        [0, 0, 20, 60, 42, 6, 0, 0],
        [0, 0, 18, 58, 44, 8, 0, 0],
        [0, 0, 16, 56, 46, 10, 0, 0],
        [0, 0, 14, 54, 48, 12, 0, 0],
        [0, 0, 12, 52, 52, 12, 0, 0],
        [0, 0, 12, 48, 54, 14, 0, 0],
        [0, 0, 10, 46, 56, 16, 0, 0],
        [0, 0, 8, 44, 58, 18, 0, 0],
        [0, 0, 6, 42, 60, 20, 0, 0],
        [0, 0, 4, 40, 62, 22, 0, 0],
        [0, 0, 4, 36, 62, 26, 0, 0],
        [0, 0, 2, 34, 62, 30, 0, 0],
    ],
];

mod rust {
    use num_traits::AsPrimitive;
    use simd_helpers::cold_for_target_arch;
    use v_frame::{math::round_shift, pixel::Pixel, plane::PlaneSlice};

    use crate::{
        cpu::CpuFeatureLevel,
        data::{
            mc::{FilterMode, SUBPEL_FILTERS, SUBPEL_FILTER_SIZE},
            plane::PlaneRegionMut,
        },
    };

    #[cfg_attr(
        all(asm_x86_64, any(target_feature = "ssse3", target_feature = "avx2")),
        cold_for_target_arch("x86_64")
    )]
    #[allow(clippy::too_many_arguments)]
    pub fn put_8tap_internal<T: Pixel>(
        dst: &mut PlaneRegionMut<'_, T>,
        src: PlaneSlice<'_, T>,
        width: usize,
        height: usize,
        col_frac: i32,
        row_frac: i32,
        bit_depth: usize,
        _cpu: CpuFeatureLevel,
    ) {
        // The assembly only supports even heights and valid uncropped widths
        assert_eq!(height & 1, 0);
        assert!(width.is_power_of_two() && (2..=128).contains(&width));

        let ref_stride = src.plane.cfg.stride;
        let y_filter = get_filter(row_frac, height);
        let x_filter = get_filter(col_frac, width);
        let max_sample_val = (1 << bit_depth) - 1;
        let intermediate_bits = 4 - if bit_depth == 12 { 2 } else { 0 };
        match (col_frac, row_frac) {
            (0, 0) => {
                for r in 0..height {
                    let src_slice = &src[r];
                    let dst_slice = &mut dst[r];
                    dst_slice[..width].copy_from_slice(&src_slice[..width]);
                }
            }
            (0, _) => {
                let offset_slice = src.go_up(3);
                for r in 0..height {
                    let src_slice = &offset_slice[r];
                    let dst_slice = &mut dst[r];
                    for c in 0..width {
                        dst_slice[c] = T::cast_from(
                            round_shift(
                                // SAFETY: We pass this a raw pointer, but it's created from a
                                // checked slice, so we are safe.
                                unsafe {
                                    run_filter(src_slice[c..].as_ptr(), ref_stride, y_filter)
                                },
                                7,
                            )
                            .clamp(0, max_sample_val),
                        );
                    }
                }
            }
            (_, 0) => {
                let offset_slice = src.go_left(3);
                for r in 0..height {
                    let src_slice = &offset_slice[r];
                    let dst_slice = &mut dst[r];
                    for c in 0..width {
                        dst_slice[c] = T::cast_from(
                            round_shift(
                                round_shift(
                                    // SAFETY: We pass this a raw pointer, but it's created from a
                                    // checked slice, so we are safe.
                                    unsafe { run_filter(src_slice[c..].as_ptr(), 1, x_filter) },
                                    7 - intermediate_bits,
                                ),
                                intermediate_bits,
                            )
                            .clamp(0, max_sample_val),
                        );
                    }
                }
            }
            (_, _) => {
                let mut intermediate: [i16; 8 * (128 + 7)] = [0; 8 * (128 + 7)];

                let offset_slice = src.go_left(3).go_up(3);
                for cg in (0..width).step_by(8) {
                    for r in 0..height + 7 {
                        let src_slice = &offset_slice[r];
                        for c in cg..(cg + 8).min(width) {
                            intermediate[8 * r + (c - cg)] = round_shift(
                                // SAFETY: We pass this a raw pointer, but it's created from a
                                // checked slice, so we are safe.
                                unsafe { run_filter(src_slice[c..].as_ptr(), 1, x_filter) },
                                7 - intermediate_bits,
                            ) as i16;
                        }
                    }

                    for r in 0..height {
                        let dst_slice = &mut dst[r];
                        for c in cg..(cg + 8).min(width) {
                            dst_slice[c] = T::cast_from(
                                round_shift(
                                    // SAFETY: We pass this a raw pointer, but it's created from a
                                    // checked slice, so we are safe.
                                    unsafe {
                                        run_filter(
                                            intermediate[8 * r + c - cg..].as_ptr(),
                                            8,
                                            y_filter,
                                        )
                                    },
                                    7 + intermediate_bits,
                                )
                                .clamp(0, max_sample_val),
                            );
                        }
                    }
                }
            }
        }
    }

    fn get_filter(frac: i32, length: usize) -> [i32; SUBPEL_FILTER_SIZE] {
        const MODE: FilterMode = FilterMode::REGULAR;

        let filter_idx = if MODE == FilterMode::BILINEAR || length > 4 {
            MODE as usize
        } else {
            (MODE as usize).min(1) + 4
        };
        SUBPEL_FILTERS[filter_idx][frac as usize]
    }

    unsafe fn run_filter<T: AsPrimitive<i32>>(
        src: *const T,
        stride: usize,
        filter: [i32; 8],
    ) -> i32 {
        filter
            .iter()
            .enumerate()
            .map(|(i, f)| {
                let p = src.add(i * stride);
                f * (*p).as_()
            })
            .sum::<i32>()
    }
}

#[cfg(asm_x86_64)]
mod simd_x86 {
    use v_frame::{
        pixel::{Pixel, PixelType},
        plane::PlaneSlice,
    };

    use crate::{
        cpu::CpuFeatureLevel,
        data::{
            mc::{FilterMode, FilterMode::*},
            plane::PlaneRegionMut,
        },
    };

    #[allow(clippy::too_many_arguments)]
    pub fn put_8tap_internal<T: Pixel>(
        dst: &mut PlaneRegionMut<'_, T>,
        src: PlaneSlice<'_, T>,
        width: usize,
        height: usize,
        col_frac: i32,
        row_frac: i32,
        bit_depth: usize,
        cpu: CpuFeatureLevel,
    ) {
        let call_rust = |dst: &mut PlaneRegionMut<'_, T>| {
            super::rust::put_8tap_internal(
                dst, src, width, height, col_frac, row_frac, bit_depth, cpu,
            );
        };

        // SAFETY: The assembly only supports even heights and valid uncropped
        //         widths
        unsafe {
            assert_eq!(height & 1, 0);
            assert!(width.is_power_of_two() && (2..=128).contains(&width));

            // SAFETY: Check bounds of dst
            assert!(dst.rect().width >= width && dst.rect().height >= height);

            // SAFETY: Check bounds of src
            assert!(src.accessible(width + 4, height + 4));
            assert!(src.accessible_neg(3, 3));

            match T::type_enum() {
                PixelType::U8 => match PUT_FNS[cpu.as_index()][get_2d_mode_idx(REGULAR, REGULAR)] {
                    Some(func) => func(
                        dst.data_ptr_mut() as *mut _,
                        T::to_asm_stride(dst.plane_cfg.stride),
                        src.as_ptr() as *const _,
                        T::to_asm_stride(src.plane.cfg.stride),
                        width as i32,
                        height as i32,
                        col_frac,
                        row_frac,
                    ),
                    None => call_rust(dst),
                },
                PixelType::U16 => {
                    match PUT_HBD_FNS[cpu.as_index()][get_2d_mode_idx(REGULAR, REGULAR)] {
                        Some(func) => func(
                            dst.data_ptr_mut() as *mut _,
                            T::to_asm_stride(dst.plane_cfg.stride),
                            src.as_ptr() as *const _,
                            T::to_asm_stride(src.plane.cfg.stride),
                            width as i32,
                            height as i32,
                            col_frac,
                            row_frac,
                            (1 << bit_depth) - 1,
                        ),
                        None => call_rust(dst),
                    }
                }
            }
        }
    }

    type PutFn = unsafe extern "C" fn(
        dst: *mut u8,
        dst_stride: isize,
        src: *const u8,
        src_stride: isize,
        width: i32,
        height: i32,
        col_frac: i32,
        row_frac: i32,
    );

    type PutHBDFn = unsafe extern "C" fn(
        dst: *mut u16,
        dst_stride: isize,
        src: *const u16,
        src_stride: isize,
        width: i32,
        height: i32,
        col_frac: i32,
        row_frac: i32,
        bitdepth_max: i32,
    );

    // gets an index that can be mapped to a function for a pair of filter modes
    const fn get_2d_mode_idx(mode_x: FilterMode, mode_y: FilterMode) -> usize {
        (mode_x as usize + 4 * (mode_y as usize)) & 15
    }

    macro_rules! decl_mc_fns {
      ($(($mode_x:expr, $mode_y:expr, $func_name:ident)),+) => {
        paste::item! {
          extern "C" {
            $(
              fn [<$func_name _ssse3>](
                dst: *mut u8, dst_stride: isize, src: *const u8, src_stride: isize,
                w: i32, h: i32, mx: i32, my: i32
              );

              fn [<$func_name _avx2>](
                dst: *mut u8, dst_stride: isize, src: *const u8, src_stride: isize,
                w: i32, h: i32, mx: i32, my: i32
              );

              fn [<$func_name _avx512icl>](
                dst: *mut u8, dst_stride: isize, src: *const u8, src_stride: isize,
                w: i32, h: i32, mx: i32, my: i32
              );
            )*
          }

          static PUT_FNS_SSSE3: [Option<PutFn>; 16] = {
            let mut out: [Option<PutFn>; 16] = [None; 16];
            $(
              out[get_2d_mode_idx($mode_x, $mode_y)] = Some([<$func_name _ssse3>]);
            )*
            out
          };

          static PUT_FNS_AVX2: [Option<PutFn>; 16] = {
            let mut out: [Option<PutFn>; 16] = [None; 16];
            $(
              out[get_2d_mode_idx($mode_x, $mode_y)] = Some([<$func_name _avx2>]);
            )*
            out
          };

          static PUT_FNS_AVX512ICL: [Option<PutFn>; 16] = {
            let mut out: [Option<PutFn>; 16] = [None; 16];
            $(
              out[get_2d_mode_idx($mode_x, $mode_y)] = Some([<$func_name _avx512icl>]);
            )*
            out
          };
        }
      }
    }

    decl_mc_fns!(
        (REGULAR, REGULAR, avsc_put_8tap_regular_8bpc),
        (REGULAR, SMOOTH, avsc_put_8tap_regular_smooth_8bpc),
        (REGULAR, SHARP, avsc_put_8tap_regular_sharp_8bpc),
        (SMOOTH, REGULAR, avsc_put_8tap_smooth_regular_8bpc),
        (SMOOTH, SMOOTH, avsc_put_8tap_smooth_8bpc),
        (SMOOTH, SHARP, avsc_put_8tap_smooth_sharp_8bpc),
        (SHARP, REGULAR, avsc_put_8tap_sharp_regular_8bpc),
        (SHARP, SMOOTH, avsc_put_8tap_sharp_smooth_8bpc),
        (SHARP, SHARP, avsc_put_8tap_sharp_8bpc),
        (BILINEAR, BILINEAR, avsc_put_bilin_8bpc)
    );

    cpu_function_lookup_table!(
      PUT_FNS: [[Option<PutFn>; 16]],
      default: [None; 16],
      [SSSE3, AVX2, AVX512ICL]
    );

    macro_rules! decl_mc_hbd_fns {
      ($(($mode_x:expr, $mode_y:expr, $func_name:ident)),+) => {
        paste::item! {
          extern "C" {
            $(
              fn [<$func_name _ssse3>](
                dst: *mut u16, dst_stride: isize, src: *const u16, src_stride: isize,
                w: i32, h: i32, mx: i32, my: i32, bitdepth_max: i32,
              );

              fn [<$func_name _avx2>](
                dst: *mut u16, dst_stride: isize, src: *const u16, src_stride: isize,
                w: i32, h: i32, mx: i32, my: i32, bitdepth_max: i32,
              );
            )*
          }

          static PUT_HBD_FNS_SSSE3: [Option<PutHBDFn>; 16] = {
            let mut out: [Option<PutHBDFn>; 16] = [None; 16];
            $(
              out[get_2d_mode_idx($mode_x, $mode_y)] = Some([<$func_name _ssse3>]);
            )*
            out
          };

          static PUT_HBD_FNS_AVX2: [Option<PutHBDFn>; 16] = {
            let mut out: [Option<PutHBDFn>; 16] = [None; 16];
            $(
              out[get_2d_mode_idx($mode_x, $mode_y)] = Some([<$func_name _avx2>]);
            )*
            out
          };
        }
      }
    }

    decl_mc_hbd_fns!(
        (REGULAR, REGULAR, avsc_put_8tap_regular_16bpc),
        (REGULAR, SMOOTH, avsc_put_8tap_regular_smooth_16bpc),
        (REGULAR, SHARP, avsc_put_8tap_regular_sharp_16bpc),
        (SMOOTH, REGULAR, avsc_put_8tap_smooth_regular_16bpc),
        (SMOOTH, SMOOTH, avsc_put_8tap_smooth_16bpc),
        (SMOOTH, SHARP, avsc_put_8tap_smooth_sharp_16bpc),
        (SHARP, REGULAR, avsc_put_8tap_sharp_regular_16bpc),
        (SHARP, SMOOTH, avsc_put_8tap_sharp_smooth_16bpc),
        (SHARP, SHARP, avsc_put_8tap_sharp_16bpc),
        (BILINEAR, BILINEAR, avsc_put_bilin_16bpc)
    );

    cpu_function_lookup_table!(
      PUT_HBD_FNS: [[Option<PutHBDFn>; 16]],
      default: [None; 16],
      [SSSE3, AVX2]
    );
}

// TODO: uncomment once code has no errors
#[cfg(asm_neon)]
mod simd_neon {
    use v_frame::{
        pixel::{Pixel, PixelType},
        plane::PlaneSlice,
    };

    use crate::{
        cpu::CpuFeatureLevel,
        data::{mc::FilterMode::*, plane::PlaneRegionMut},
    };

    pub fn put_8tap_internal<T: Pixel>(
        dst: &mut PlaneRegionMut<'_, T>,
        src: PlaneSlice<'_, T>,
        width: usize,
        height: usize,
        col_frac: i32,
        row_frac: i32,
        bit_depth: usize,
        cpu: CpuFeatureLevel,
    ) {
        let call_rust = |dst: &mut PlaneRegionMut<'_, T>| {
            super::rust::put_8tap_internal(
                dst, src, width, height, col_frac, row_frac, bit_depth, cpu,
            );
        };

        unsafe {
            // SAFETY: The assembly only supports even heights and valid uncropped
            //         widths
            assert_eq!(height & 1, 0);
            assert!(width.is_power_of_two() && (2..=128).contains(&width));

            // SAFETY: Check bounds of dst
            assert!(dst.rect().width >= width && dst.rect().height >= height);

            // SAFETY: Check bounds of src
            assert!(src.accessible(width + 4, height + 4));
            assert!(src.accessible_neg(3, 3));

            match T::type_enum() {
                PixelType::U8 => match PUT_FNS[cpu.as_index()]
                    [get_2d_mode_idx(FilterMode::REGULAR, FilterMode::REGULAR)]
                {
                    Some(func) => (func)(
                        dst.data_ptr_mut() as *mut _,
                        T::to_asm_stride(dst.plane_cfg.stride),
                        src.as_ptr() as *const _,
                        T::to_asm_stride(src.plane.cfg.stride),
                        width as i32,
                        height as i32,
                        col_frac,
                        row_frac,
                    ),
                    None => call_rust(dst),
                },
                PixelType::U16 if bit_depth > 8 => {
                    match PUT_HBD_FNS[cpu.as_index()]
                        [get_2d_mode_idx(FilterMode::REGULAR, FilterMode::REGULAR)]
                    {
                        Some(func) => (func)(
                            dst.data_ptr_mut() as *mut _,
                            T::to_asm_stride(dst.plane_cfg.stride),
                            src.as_ptr() as *const _,
                            T::to_asm_stride(src.plane.cfg.stride),
                            width as i32,
                            height as i32,
                            col_frac,
                            row_frac,
                            (1 << bit_depth) - 1,
                        ),
                        None => call_rust(dst),
                    }
                }
                _ => call_rust(dst),
            }
        }
    }

    // gets an index that can be mapped to a function for a pair of filter modes
    const fn get_2d_mode_idx(mode_x: FilterMode, mode_y: FilterMode) -> usize {
        (mode_x as usize + 4 * (mode_y as usize)) & 15
    }

    type PutFn = unsafe extern "C" fn(
        dst: *mut u8,
        dst_stride: isize,
        src: *const u8,
        src_stride: isize,
        width: i32,
        height: i32,
        col_frac: i32,
        row_frac: i32,
    );

    type PutHBDFn = unsafe extern "C" fn(
        dst: *mut u16,
        dst_stride: isize,
        src: *const u16,
        src_stride: isize,
        width: i32,
        height: i32,
        col_frac: i32,
        row_frac: i32,
        bitdepth_max: i32,
    );

    macro_rules! decl_mc_fns {
      ($(($mode_x:expr, $mode_y:expr, $func_name:ident)),+) => {
        extern {
          $(
            fn $func_name(
              dst: *mut u8, dst_stride: isize, src: *const u8, src_stride: isize,
              w: i32, h: i32, mx: i32, my: i32,
            );
          )*
        }

        static PUT_FNS_NEON: [Option<PutFn>; 16] = {
          let mut out: [Option<PutFn>; 16] = [None; 16];
          $(
            out[get_2d_mode_idx($mode_x, $mode_y)] = Some($func_name);
          )*
          out
        };
      }
    }

    decl_mc_fns!(
        (REGULAR, REGULAR, avsc_put_8tap_regular_8bpc_neon),
        (REGULAR, SMOOTH, avsc_put_8tap_regular_smooth_8bpc_neon),
        (REGULAR, SHARP, avsc_put_8tap_regular_sharp_8bpc_neon),
        (SMOOTH, REGULAR, avsc_put_8tap_smooth_regular_8bpc_neon),
        (SMOOTH, SMOOTH, avsc_put_8tap_smooth_8bpc_neon),
        (SMOOTH, SHARP, avsc_put_8tap_smooth_sharp_8bpc_neon),
        (SHARP, REGULAR, avsc_put_8tap_sharp_regular_8bpc_neon),
        (SHARP, SMOOTH, avsc_put_8tap_sharp_smooth_8bpc_neon),
        (SHARP, SHARP, avsc_put_8tap_sharp_8bpc_neon),
        (BILINEAR, BILINEAR, avsc_put_bilin_8bpc_neon)
    );

    cpu_function_lookup_table!(
      PUT_FNS: [[Option<PutFn>; 16]],
      default: [None; 16],
      [NEON]
    );

    macro_rules! decl_mc_hbd_fns {
      ($(($mode_x:expr, $mode_y:expr, $func_name:ident)),+) => {
        extern {
          $(
            fn $func_name(
              dst: *mut u16, dst_stride: isize, src: *const u16, src_stride: isize,
              w: i32, h: i32, mx: i32, my: i32, bitdepth_max: i32,
            );
          )*
        }

        static PUT_HBD_FNS_NEON: [Option<PutHBDFn>; 16] = {
          let mut out: [Option<PutHBDFn>; 16] = [None; 16];
          $(
            out[get_2d_mode_idx($mode_x, $mode_y)] = Some($func_name);
          )*
          out
        };
      }
    }

    decl_mc_hbd_fns!(
        (REGULAR, REGULAR, avsc_put_8tap_regular_16bpc_neon),
        (REGULAR, SMOOTH, avsc_put_8tap_regular_smooth_16bpc_neon),
        (REGULAR, SHARP, avsc_put_8tap_regular_sharp_16bpc_neon),
        (SMOOTH, REGULAR, avsc_put_8tap_smooth_regular_16bpc_neon),
        (SMOOTH, SMOOTH, avsc_put_8tap_smooth_16bpc_neon),
        (SMOOTH, SHARP, avsc_put_8tap_smooth_sharp_16bpc_neon),
        (SHARP, REGULAR, avsc_put_8tap_sharp_regular_16bpc_neon),
        (SHARP, SMOOTH, avsc_put_8tap_sharp_smooth_16bpc_neon),
        (SHARP, SHARP, avsc_put_8tap_sharp_16bpc_neon),
        (BILINEAR, BILINEAR, avsc_put_bilin_16bpc_neon)
    );

    cpu_function_lookup_table!(
      PUT_HBD_FNS: [[Option<PutHBDFn>; 16]],
      default: [None; 16],
      [NEON]
    );
}

#[allow(clippy::too_many_arguments)]
pub fn put_8tap<T: Pixel>(
    dst: &mut PlaneRegionMut<'_, T>,
    src: PlaneSlice<'_, T>,
    width: usize,
    height: usize,
    col_frac: i32,
    row_frac: i32,
    bit_depth: usize,
    cpu: CpuFeatureLevel,
) {
    put_8tap_internal(dst, src, width, height, col_frac, row_frac, bit_depth, cpu);
}
