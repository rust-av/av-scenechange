#[cfg(test)]
mod tests;

#[cfg(not(any(asm_x86_64, asm_neon)))]
use rust::*;
#[cfg(asm_neon)]
use simd_neon::*;
#[cfg(asm_x86_64)]
use simd_x86::*;
use v_frame::pixel::Pixel;

use super::{block::BlockSize, plane::PlaneRegion};
use crate::cpu::CpuFeatureLevel;

mod rust {
    use simd_helpers::cold_for_target_arch;
    use v_frame::{
        math::msb,
        pixel::{CastFromPrimitive, Pixel},
    };

    use crate::{
        cpu::CpuFeatureLevel,
        data::{
            hadamard::{hadamard4x4, hadamard8x8},
            plane::{Area, PlaneRegion, Rect},
            sad::get_sad,
        },
    };

    /// Sum of absolute transformed differences over a block.
    /// w and h can be at most 128, the size of the largest block.
    /// Use the sum of 4x4 and 8x8 hadamard transforms for the transform, but
    /// revert to sad on edges when these transforms do not fit into w and h.
    /// 4x4 transforms instead of 8x8 transforms when width or height < 8.
    #[cfg_attr(
        all(asm_x86_64, target_feature = "avx2"),
        cold_for_target_arch("x86_64")
    )]
    pub(super) fn get_satd_internal<T: Pixel>(
        plane_org: &PlaneRegion<'_, T>,
        plane_ref: &PlaneRegion<'_, T>,
        w: usize,
        h: usize,
        bit_depth: usize,
        cpu: CpuFeatureLevel,
    ) -> u32 {
        assert!(w <= 128 && h <= 128);
        assert!(plane_org.rect().width >= w && plane_org.rect().height >= h);
        assert!(plane_ref.rect().width >= w && plane_ref.rect().height >= h);

        // Size of hadamard transform should be 4x4 or 8x8
        // 4x* and *x4 use 4x4 and all other use 8x8
        let size: usize = w.min(h).min(8);
        let tx2d = if size == 4 { hadamard4x4 } else { hadamard8x8 };

        let mut sum: u64 = 0;

        // Loop over chunks the size of the chosen transform
        for chunk_y in (0..h).step_by(size) {
            let chunk_h = (h - chunk_y).min(size);
            for chunk_x in (0..w).step_by(size) {
                let chunk_w = (w - chunk_x).min(size);
                let chunk_area = Area::Rect(Rect {
                    x: chunk_x as isize,
                    y: chunk_y as isize,
                    width: chunk_w,
                    height: chunk_h,
                });
                let chunk_org = plane_org.subregion(chunk_area);
                let chunk_ref = plane_ref.subregion(chunk_area);

                // Revert to sad on edge blocks (frame edges)
                if chunk_w != size || chunk_h != size {
                    sum += get_sad(&chunk_org, &chunk_ref, chunk_w, chunk_h, bit_depth, cpu) as u64;
                    continue;
                }

                let buf: &mut [i32] = &mut [0; 8 * 8][..size * size];

                // Move the difference of the transforms to a buffer
                for (row_diff, (row_org, row_ref)) in buf
                    .chunks_mut(size)
                    .zip(chunk_org.rows_iter().zip(chunk_ref.rows_iter()))
                {
                    for (diff, (a, b)) in
                        row_diff.iter_mut().zip(row_org.iter().zip(row_ref.iter()))
                    {
                        *diff = i32::cast_from(*a) - i32::cast_from(*b);
                    }
                }

                // Perform the hadamard transform on the differences
                // SAFETY: A sufficient number elements exist for the size of the transform.
                unsafe {
                    tx2d(buf);
                }

                // Sum the absolute values of the transformed differences
                sum += buf.iter().map(|a| a.unsigned_abs() as u64).sum::<u64>();
            }
        }

        // Normalize the results
        let ln = msb(size as i32) as u64;
        ((sum + (1 << ln >> 1)) >> ln) as u32
    }
}

#[cfg(asm_x86_64)]
mod simd_x86 {
    use v_frame::pixel::{Pixel, PixelType};

    use super::{to_index, DIST_FNS_LENGTH};
    use crate::{
        cpu::CpuFeatureLevel,
        data::{block::BlockSize, plane::PlaneRegion},
    };

    type SatdFn = unsafe extern "C" fn(
        src: *const u8,
        src_stride: isize,
        dst: *const u8,
        dst_stride: isize,
    ) -> u32;

    type SatdHBDFn = unsafe extern "C" fn(
        src: *const u16,
        src_stride: isize,
        dst: *const u16,
        dst_stride: isize,
        bdmax: u32,
    ) -> u32;

    macro_rules! declare_asm_dist_fn {
        ($(($name: ident, $T: ident)),+) => (
            $(
            extern "C" { fn $name (
                src: *const $T, src_stride: isize, dst: *const $T, dst_stride: isize
            ) -> u32; }
            )+
        )
    }

    macro_rules! declare_asm_satd_hbd_fn {
        ($($name: ident),+) => (
            $(
            extern "C" { pub(crate) fn $name (
                src: *const u16, src_stride: isize, dst: *const u16, dst_stride: isize, bdmax: u32
            ) -> u32; }
            )+
        )
    }

    declare_asm_dist_fn![
        // SSSE3
        (avsc_satd_8x8_ssse3, u8),
        // SSE4
        (avsc_satd_4x4_sse4, u8),
        // AVX
        (avsc_satd_4x4_avx2, u8),
        (avsc_satd_8x8_avx2, u8),
        (avsc_satd_16x16_avx2, u8),
        (avsc_satd_32x32_avx2, u8),
        (avsc_satd_64x64_avx2, u8),
        (avsc_satd_128x128_avx2, u8),
        (avsc_satd_4x8_avx2, u8),
        (avsc_satd_8x4_avx2, u8),
        (avsc_satd_8x16_avx2, u8),
        (avsc_satd_16x8_avx2, u8),
        (avsc_satd_16x32_avx2, u8),
        (avsc_satd_32x16_avx2, u8),
        (avsc_satd_32x64_avx2, u8),
        (avsc_satd_64x32_avx2, u8),
        (avsc_satd_64x128_avx2, u8),
        (avsc_satd_128x64_avx2, u8),
        (avsc_satd_4x16_avx2, u8),
        (avsc_satd_16x4_avx2, u8),
        (avsc_satd_8x32_avx2, u8),
        (avsc_satd_32x8_avx2, u8),
        (avsc_satd_16x64_avx2, u8),
        (avsc_satd_64x16_avx2, u8)
    ];

    declare_asm_satd_hbd_fn![
        avsc_satd_4x4_hbd_avx2,
        avsc_satd_8x4_hbd_avx2,
        avsc_satd_4x8_hbd_avx2,
        avsc_satd_8x8_hbd_avx2,
        avsc_satd_16x8_hbd_avx2,
        avsc_satd_16x16_hbd_avx2,
        avsc_satd_32x32_hbd_avx2,
        avsc_satd_64x64_hbd_avx2,
        avsc_satd_128x128_hbd_avx2,
        avsc_satd_16x32_hbd_avx2,
        avsc_satd_16x64_hbd_avx2,
        avsc_satd_32x16_hbd_avx2,
        avsc_satd_32x64_hbd_avx2,
        avsc_satd_64x16_hbd_avx2,
        avsc_satd_64x32_hbd_avx2,
        avsc_satd_64x128_hbd_avx2,
        avsc_satd_128x64_hbd_avx2,
        avsc_satd_32x8_hbd_avx2,
        avsc_satd_8x16_hbd_avx2,
        avsc_satd_8x32_hbd_avx2,
        avsc_satd_16x4_hbd_avx2,
        avsc_satd_4x16_hbd_avx2
    ];

    static SATD_FNS_SSSE3: [Option<SatdFn>; DIST_FNS_LENGTH] = {
        let mut out: [Option<SatdFn>; DIST_FNS_LENGTH] = [None; DIST_FNS_LENGTH];

        use BlockSize::*;

        out[BLOCK_8X8 as usize] = Some(avsc_satd_8x8_ssse3);

        out
    };

    static SATD_FNS_SSE4_1: [Option<SatdFn>; DIST_FNS_LENGTH] = {
        let mut out: [Option<SatdFn>; DIST_FNS_LENGTH] = [None; DIST_FNS_LENGTH];

        use BlockSize::*;

        out[BLOCK_4X4 as usize] = Some(avsc_satd_4x4_sse4);
        out[BLOCK_8X8 as usize] = Some(avsc_satd_8x8_ssse3);

        out
    };

    static SATD_FNS_AVX2: [Option<SatdFn>; DIST_FNS_LENGTH] = {
        let mut out: [Option<SatdFn>; DIST_FNS_LENGTH] = [None; DIST_FNS_LENGTH];

        use BlockSize::*;

        out[BLOCK_4X4 as usize] = Some(avsc_satd_4x4_avx2);
        out[BLOCK_8X8 as usize] = Some(avsc_satd_8x8_avx2);
        out[BLOCK_16X16 as usize] = Some(avsc_satd_16x16_avx2);
        out[BLOCK_32X32 as usize] = Some(avsc_satd_32x32_avx2);
        out[BLOCK_64X64 as usize] = Some(avsc_satd_64x64_avx2);
        out[BLOCK_128X128 as usize] = Some(avsc_satd_128x128_avx2);

        out[BLOCK_4X8 as usize] = Some(avsc_satd_4x8_avx2);
        out[BLOCK_8X4 as usize] = Some(avsc_satd_8x4_avx2);
        out[BLOCK_8X16 as usize] = Some(avsc_satd_8x16_avx2);
        out[BLOCK_16X8 as usize] = Some(avsc_satd_16x8_avx2);
        out[BLOCK_16X32 as usize] = Some(avsc_satd_16x32_avx2);
        out[BLOCK_32X16 as usize] = Some(avsc_satd_32x16_avx2);
        out[BLOCK_32X64 as usize] = Some(avsc_satd_32x64_avx2);
        out[BLOCK_64X32 as usize] = Some(avsc_satd_64x32_avx2);
        out[BLOCK_64X128 as usize] = Some(avsc_satd_64x128_avx2);
        out[BLOCK_128X64 as usize] = Some(avsc_satd_128x64_avx2);

        out[BLOCK_4X16 as usize] = Some(avsc_satd_4x16_avx2);
        out[BLOCK_16X4 as usize] = Some(avsc_satd_16x4_avx2);
        out[BLOCK_8X32 as usize] = Some(avsc_satd_8x32_avx2);
        out[BLOCK_32X8 as usize] = Some(avsc_satd_32x8_avx2);
        out[BLOCK_16X64 as usize] = Some(avsc_satd_16x64_avx2);
        out[BLOCK_64X16 as usize] = Some(avsc_satd_64x16_avx2);

        out
    };

    cpu_function_lookup_table!(
      SATD_FNS: [[Option<SatdFn>; DIST_FNS_LENGTH]],
      default: [None; DIST_FNS_LENGTH],
      [SSSE3, SSE4_1, AVX2]
    );

    static SATD_HBD_FNS_AVX2: [Option<SatdHBDFn>; DIST_FNS_LENGTH] = {
        let mut out: [Option<SatdHBDFn>; DIST_FNS_LENGTH] = [None; DIST_FNS_LENGTH];

        use BlockSize::*;

        out[BLOCK_4X4 as usize] = Some(avsc_satd_4x4_hbd_avx2);
        out[BLOCK_8X8 as usize] = Some(avsc_satd_8x8_hbd_avx2);
        out[BLOCK_16X16 as usize] = Some(avsc_satd_16x16_hbd_avx2);
        out[BLOCK_32X32 as usize] = Some(avsc_satd_32x32_hbd_avx2);
        out[BLOCK_64X64 as usize] = Some(avsc_satd_64x64_hbd_avx2);
        out[BLOCK_128X128 as usize] = Some(avsc_satd_128x128_hbd_avx2);

        out[BLOCK_4X8 as usize] = Some(avsc_satd_4x8_hbd_avx2);
        out[BLOCK_8X4 as usize] = Some(avsc_satd_8x4_hbd_avx2);
        out[BLOCK_8X16 as usize] = Some(avsc_satd_8x16_hbd_avx2);
        out[BLOCK_16X8 as usize] = Some(avsc_satd_16x8_hbd_avx2);
        out[BLOCK_16X32 as usize] = Some(avsc_satd_16x32_hbd_avx2);
        out[BLOCK_32X16 as usize] = Some(avsc_satd_32x16_hbd_avx2);
        out[BLOCK_32X64 as usize] = Some(avsc_satd_32x64_hbd_avx2);
        out[BLOCK_64X32 as usize] = Some(avsc_satd_64x32_hbd_avx2);
        out[BLOCK_64X128 as usize] = Some(avsc_satd_64x128_hbd_avx2);
        out[BLOCK_128X64 as usize] = Some(avsc_satd_128x64_hbd_avx2);

        out[BLOCK_4X16 as usize] = Some(avsc_satd_4x16_hbd_avx2);
        out[BLOCK_16X4 as usize] = Some(avsc_satd_16x4_hbd_avx2);
        out[BLOCK_8X32 as usize] = Some(avsc_satd_8x32_hbd_avx2);
        out[BLOCK_32X8 as usize] = Some(avsc_satd_32x8_hbd_avx2);
        out[BLOCK_16X64 as usize] = Some(avsc_satd_16x64_hbd_avx2);
        out[BLOCK_64X16 as usize] = Some(avsc_satd_64x16_hbd_avx2);

        out
    };

    cpu_function_lookup_table!(
      SATD_HBD_FNS: [[Option<SatdHBDFn>; DIST_FNS_LENGTH]],
      default: [None; DIST_FNS_LENGTH],
      [AVX2]
    );

    pub(super) fn get_satd_internal<T: Pixel>(
        src: &PlaneRegion<'_, T>,
        dst: &PlaneRegion<'_, T>,
        w: usize,
        h: usize,
        bit_depth: usize,
        cpu: CpuFeatureLevel,
    ) -> u32 {
        let bsize_opt = BlockSize::from_width_and_height_opt(w, h);

        let call_rust =
            || -> u32 { super::rust::get_satd_internal(dst, src, w, h, bit_depth, cpu) };

        match (bsize_opt, T::type_enum()) {
            (Err(_), _) => call_rust(),
            (Ok(bsize), PixelType::U8) => {
                match SATD_FNS[cpu.as_index()][to_index(bsize)] {
                    // SAFETY: Calls Assembly code.
                    Some(func) => unsafe {
                        func(
                            src.data_ptr() as *const _,
                            T::to_asm_stride(src.plane_cfg.stride),
                            dst.data_ptr() as *const _,
                            T::to_asm_stride(dst.plane_cfg.stride),
                        )
                    },
                    None => call_rust(),
                }
            }
            (Ok(bsize), PixelType::U16) => {
                match SATD_HBD_FNS[cpu.as_index()][to_index(bsize)] {
                    // SAFETY: Calls Assembly code.
                    Some(func) => unsafe {
                        func(
                            src.data_ptr() as *const _,
                            T::to_asm_stride(src.plane_cfg.stride),
                            dst.data_ptr() as *const _,
                            T::to_asm_stride(dst.plane_cfg.stride),
                            (1 << bit_depth) - 1,
                        )
                    },
                    None => call_rust(),
                }
            }
        }
    }
}

// TODO: uncomment once code has no errors
#[cfg(asm_neon)]
mod simd_neon {
    use v_frame::pixel::{Pixel, PixelType};

    use super::{to_index, DIST_FNS_LENGTH};
    use crate::{
        cpu::CpuFeatureLevel,
        data::{block::BlockSize, plane::PlaneRegion},
    };

    type SatdFn = unsafe extern "C" fn(
        src: *const u8,
        src_stride: isize,
        dst: *const u8,
        dst_stride: isize,
    ) -> u32;
    type SatdHbdFn = unsafe extern "C" fn(
        src: *const u16,
        src_stride: isize,
        dst: *const u16,
        dst_stride: isize,
    ) -> u32;

    macro_rules! declare_asm_dist_fn {
        ($(($name: ident, $T: ident)),+) => (
            $(
                extern "C" { fn $name (
                    src: *const $T, src_stride: isize, dst: *const $T, dst_stride: isize
                ) -> u32; }
            )+
        )
    }

    declare_asm_dist_fn![
        // SATD
        (avsc_satd4x4_neon, u8),
        (avsc_satd4x8_neon, u8),
        (avsc_satd4x16_neon, u8),
        (avsc_satd8x4_neon, u8),
        (avsc_satd8x8_neon, u8),
        (avsc_satd8x16_neon, u8),
        (avsc_satd8x32_neon, u8),
        (avsc_satd16x4_neon, u8),
        (avsc_satd16x8_neon, u8),
        (avsc_satd16x16_neon, u8),
        (avsc_satd16x32_neon, u8),
        (avsc_satd16x64_neon, u8),
        (avsc_satd32x8_neon, u8),
        (avsc_satd32x16_neon, u8),
        (avsc_satd32x32_neon, u8),
        (avsc_satd32x64_neon, u8),
        (avsc_satd64x16_neon, u8),
        (avsc_satd64x32_neon, u8),
        (avsc_satd64x64_neon, u8),
        (avsc_satd64x128_neon, u8),
        (avsc_satd128x64_neon, u8),
        (avsc_satd128x128_neon, u8),
        // SATD HBD
        (avsc_satd4x4_hbd_neon, u16),
        (avsc_satd4x8_hbd_neon, u16),
        (avsc_satd4x16_hbd_neon, u16),
        (avsc_satd8x4_hbd_neon, u16),
        (avsc_satd8x8_hbd_neon, u16),
        (avsc_satd8x16_hbd_neon, u16),
        (avsc_satd8x32_hbd_neon, u16),
        (avsc_satd16x4_hbd_neon, u16),
        (avsc_satd16x8_hbd_neon, u16),
        (avsc_satd16x16_hbd_neon, u16),
        (avsc_satd16x32_hbd_neon, u16),
        (avsc_satd16x64_hbd_neon, u16),
        (avsc_satd32x8_hbd_neon, u16),
        (avsc_satd32x16_hbd_neon, u16),
        (avsc_satd32x32_hbd_neon, u16),
        (avsc_satd32x64_hbd_neon, u16),
        (avsc_satd64x16_hbd_neon, u16),
        (avsc_satd64x32_hbd_neon, u16),
        (avsc_satd64x64_hbd_neon, u16),
        (avsc_satd64x128_hbd_neon, u16),
        (avsc_satd128x64_hbd_neon, u16),
        (avsc_satd128x128_hbd_neon, u16)
    ];

    static SATD_FNS_NEON: [Option<SatdFn>; DIST_FNS_LENGTH] = {
        let mut out: [Option<SatdFn>; DIST_FNS_LENGTH] = [None; DIST_FNS_LENGTH];

        use crate::data::block::BlockSize::*;

        out[BLOCK_4X4 as usize] = Some(avsc_satd4x4_neon);
        out[BLOCK_4X8 as usize] = Some(avsc_satd4x8_neon);
        out[BLOCK_4X16 as usize] = Some(avsc_satd4x16_neon);
        out[BLOCK_8X4 as usize] = Some(avsc_satd8x4_neon);
        out[BLOCK_16X4 as usize] = Some(avsc_satd16x4_neon);

        out[BLOCK_8X8 as usize] = Some(avsc_satd8x8_neon);
        out[BLOCK_8X16 as usize] = Some(avsc_satd8x16_neon);
        out[BLOCK_8X32 as usize] = Some(avsc_satd8x32_neon);
        out[BLOCK_16X8 as usize] = Some(avsc_satd16x8_neon);
        out[BLOCK_16X16 as usize] = Some(avsc_satd16x16_neon);
        out[BLOCK_16X32 as usize] = Some(avsc_satd16x32_neon);
        out[BLOCK_16X64 as usize] = Some(avsc_satd16x64_neon);
        out[BLOCK_32X8 as usize] = Some(avsc_satd32x8_neon);
        out[BLOCK_32X16 as usize] = Some(avsc_satd32x16_neon);
        out[BLOCK_32X32 as usize] = Some(avsc_satd32x32_neon);
        out[BLOCK_32X64 as usize] = Some(avsc_satd32x64_neon);
        out[BLOCK_64X16 as usize] = Some(avsc_satd64x16_neon);
        out[BLOCK_64X32 as usize] = Some(avsc_satd64x32_neon);
        out[BLOCK_64X64 as usize] = Some(avsc_satd64x64_neon);
        out[BLOCK_64X128 as usize] = Some(avsc_satd64x128_neon);
        out[BLOCK_128X64 as usize] = Some(avsc_satd128x64_neon);
        out[BLOCK_128X128 as usize] = Some(avsc_satd128x128_neon);

        out
    };

    static SATD_HBD_FNS_NEON: [Option<SatdHbdFn>; DIST_FNS_LENGTH] = {
        let mut out: [Option<SatdHbdFn>; DIST_FNS_LENGTH] = [None; DIST_FNS_LENGTH];

        use crate::data::block::BlockSize::*;

        out[BLOCK_4X4 as usize] = Some(avsc_satd4x4_hbd_neon);
        out[BLOCK_4X8 as usize] = Some(avsc_satd4x8_hbd_neon);
        out[BLOCK_4X16 as usize] = Some(avsc_satd4x16_hbd_neon);
        out[BLOCK_8X4 as usize] = Some(avsc_satd8x4_hbd_neon);
        out[BLOCK_16X4 as usize] = Some(avsc_satd16x4_hbd_neon);

        out[BLOCK_8X8 as usize] = Some(avsc_satd8x8_hbd_neon);
        out[BLOCK_8X16 as usize] = Some(avsc_satd8x16_hbd_neon);
        out[BLOCK_8X32 as usize] = Some(avsc_satd8x32_hbd_neon);
        out[BLOCK_16X8 as usize] = Some(avsc_satd16x8_hbd_neon);
        out[BLOCK_16X16 as usize] = Some(avsc_satd16x16_hbd_neon);
        out[BLOCK_16X32 as usize] = Some(avsc_satd16x32_hbd_neon);
        out[BLOCK_16X64 as usize] = Some(avsc_satd16x64_hbd_neon);
        out[BLOCK_32X8 as usize] = Some(avsc_satd32x8_hbd_neon);
        out[BLOCK_32X16 as usize] = Some(avsc_satd32x16_hbd_neon);
        out[BLOCK_32X32 as usize] = Some(avsc_satd32x32_hbd_neon);
        out[BLOCK_32X64 as usize] = Some(avsc_satd32x64_hbd_neon);
        out[BLOCK_64X16 as usize] = Some(avsc_satd64x16_hbd_neon);
        out[BLOCK_64X32 as usize] = Some(avsc_satd64x32_hbd_neon);
        out[BLOCK_64X64 as usize] = Some(avsc_satd64x64_hbd_neon);
        out[BLOCK_64X128 as usize] = Some(avsc_satd64x128_hbd_neon);
        out[BLOCK_128X64 as usize] = Some(avsc_satd128x64_hbd_neon);
        out[BLOCK_128X128 as usize] = Some(avsc_satd128x128_hbd_neon);

        out
    };

    cpu_function_lookup_table!(
      SATD_FNS: [[Option<SatdFn>; DIST_FNS_LENGTH]],
      default: [None; DIST_FNS_LENGTH],
      [NEON]
    );

    cpu_function_lookup_table!(
      SATD_HBD_FNS: [[Option<SatdHbdFn>; DIST_FNS_LENGTH]],
      default: [None; DIST_FNS_LENGTH],
      [NEON]
    );

    pub(super) fn get_satd_internal<T: Pixel>(
        src: &PlaneRegion<'_, T>,
        dst: &PlaneRegion<'_, T>,
        w: usize,
        h: usize,
        bit_depth: usize,
        cpu: CpuFeatureLevel,
    ) -> u32 {
        let bsize_opt = BlockSize::from_width_and_height_opt(w, h);

        let call_rust =
            || -> u32 { super::rust::get_satd_internal(src, dst, w, h, bit_depth, cpu) };

        let dist = match (bsize_opt, T::type_enum()) {
            (Err(_), _) => call_rust(),
            (Ok(bsize), PixelType::U8) => {
                match SATD_FNS[cpu.as_index()][to_index(bsize)] {
                    // SAFETY: Calls Assembly code.
                    Some(func) => unsafe {
                        (func)(
                            src.data_ptr() as *const _,
                            T::to_asm_stride(src.plane_cfg.stride),
                            dst.data_ptr() as *const _,
                            T::to_asm_stride(dst.plane_cfg.stride),
                        )
                    },
                    None => call_rust(),
                }
            }
            (Ok(bsize), PixelType::U16) => {
                match SATD_HBD_FNS[cpu.as_index()][to_index(bsize)] {
                    // SAFETY: Calls Assembly code.
                    Some(func) => unsafe {
                        (func)(
                            src.data_ptr() as *const _,
                            T::to_asm_stride(src.plane_cfg.stride),
                            dst.data_ptr() as *const _,
                            T::to_asm_stride(dst.plane_cfg.stride),
                        )
                    },
                    None => call_rust(),
                }
            }
        };

        dist
    }
}

// BlockSize::BLOCK_SIZES.next_power_of_two()
const DIST_FNS_LENGTH: usize = 32;

const fn to_index(bsize: BlockSize) -> usize {
    bsize as usize & (DIST_FNS_LENGTH - 1)
}

pub(crate) fn get_satd<T: Pixel>(
    src: &PlaneRegion<'_, T>,
    dst: &PlaneRegion<'_, T>,
    w: usize,
    h: usize,
    bit_depth: usize,
    cpu: CpuFeatureLevel,
) -> u32 {
    get_satd_internal(src, dst, w, h, bit_depth, cpu)
}
