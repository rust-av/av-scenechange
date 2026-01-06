use v_frame::pixel::Pixel;

use crate::data::{block::BlockSize, plane::PlaneRegion};

macro_rules! declare_asm_dist_fn {
        ($(($name: ident, $T: ident)),+) => (
            $(
                unsafe extern "C" { fn $name (
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

#[target_feature(enable = "neon")]
pub(super) fn get_satd_internal<T: Pixel>(
    src: &PlaneRegion<'_, T>,
    dst: &PlaneRegion<'_, T>,
    w: usize,
    h: usize,
    bit_depth: usize,
) -> u32 {
    let bsize_opt = BlockSize::from_width_and_height_opt(w, h);

    match (bsize_opt, size_of::<T>()) {
        (Err(_), _) => super::rust::get_satd_internal(src, dst, w, h, bit_depth),
        // SAFETY: call to SIMD function
        (Ok(bsize), 1) => unsafe {
            (match bsize {
                BlockSize::BLOCK_4X4 => avsc_satd4x4_neon,
                BlockSize::BLOCK_4X8 => avsc_satd4x8_neon,
                BlockSize::BLOCK_4X16 => avsc_satd4x16_neon,
                BlockSize::BLOCK_8X4 => avsc_satd8x4_neon,
                BlockSize::BLOCK_16X4 => avsc_satd16x4_neon,
                BlockSize::BLOCK_8X8 => avsc_satd8x8_neon,
                BlockSize::BLOCK_8X16 => avsc_satd8x16_neon,
                BlockSize::BLOCK_8X32 => avsc_satd8x32_neon,
                BlockSize::BLOCK_16X8 => avsc_satd16x8_neon,
                BlockSize::BLOCK_16X16 => avsc_satd16x16_neon,
                BlockSize::BLOCK_16X32 => avsc_satd16x32_neon,
                BlockSize::BLOCK_16X64 => avsc_satd16x64_neon,
                BlockSize::BLOCK_32X8 => avsc_satd32x8_neon,
                BlockSize::BLOCK_32X16 => avsc_satd32x16_neon,
                BlockSize::BLOCK_32X32 => avsc_satd32x32_neon,
                BlockSize::BLOCK_32X64 => avsc_satd32x64_neon,
                BlockSize::BLOCK_64X16 => avsc_satd64x16_neon,
                BlockSize::BLOCK_64X32 => avsc_satd64x32_neon,
                BlockSize::BLOCK_64X64 => avsc_satd64x64_neon,
                BlockSize::BLOCK_64X128 => avsc_satd64x128_neon,
                BlockSize::BLOCK_128X64 => avsc_satd128x64_neon,
                BlockSize::BLOCK_128X128 => avsc_satd128x128_neon,
            })(
                src.data_ptr() as *const _,
                (size_of::<T>() * src.plane_cfg.stride.get()) as isize,
                dst.data_ptr() as *const _,
                (size_of::<T>() * dst.plane_cfg.stride.get()) as isize,
            )
        },
        // SAFETY: call to SIMD function
        (Ok(bsize), 2) => unsafe {
            (match bsize {
                BlockSize::BLOCK_4X4 => avsc_satd4x4_hbd_neon,
                BlockSize::BLOCK_4X8 => avsc_satd4x8_hbd_neon,
                BlockSize::BLOCK_4X16 => avsc_satd4x16_hbd_neon,
                BlockSize::BLOCK_8X4 => avsc_satd8x4_hbd_neon,
                BlockSize::BLOCK_16X4 => avsc_satd16x4_hbd_neon,
                BlockSize::BLOCK_8X8 => avsc_satd8x8_hbd_neon,
                BlockSize::BLOCK_8X16 => avsc_satd8x16_hbd_neon,
                BlockSize::BLOCK_8X32 => avsc_satd8x32_hbd_neon,
                BlockSize::BLOCK_16X8 => avsc_satd16x8_hbd_neon,
                BlockSize::BLOCK_16X16 => avsc_satd16x16_hbd_neon,
                BlockSize::BLOCK_16X32 => avsc_satd16x32_hbd_neon,
                BlockSize::BLOCK_16X64 => avsc_satd16x64_hbd_neon,
                BlockSize::BLOCK_32X8 => avsc_satd32x8_hbd_neon,
                BlockSize::BLOCK_32X16 => avsc_satd32x16_hbd_neon,
                BlockSize::BLOCK_32X32 => avsc_satd32x32_hbd_neon,
                BlockSize::BLOCK_32X64 => avsc_satd32x64_hbd_neon,
                BlockSize::BLOCK_64X16 => avsc_satd64x16_hbd_neon,
                BlockSize::BLOCK_64X32 => avsc_satd64x32_hbd_neon,
                BlockSize::BLOCK_64X64 => avsc_satd64x64_hbd_neon,
                BlockSize::BLOCK_64X128 => avsc_satd64x128_hbd_neon,
                BlockSize::BLOCK_128X64 => avsc_satd128x64_hbd_neon,
                BlockSize::BLOCK_128X128 => avsc_satd128x128_hbd_neon,
            })(
                src.data_ptr() as *const _,
                (size_of::<T>() * src.plane_cfg.stride.get()) as isize,
                dst.data_ptr() as *const _,
                (size_of::<T>() * dst.plane_cfg.stride.get()) as isize,
            )
        },
        _ => unreachable!(),
    }
}
