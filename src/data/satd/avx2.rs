use v_frame::pixel::{Pixel, PixelType};

use crate::data::{block::BlockSize, plane::PlaneRegion};

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
    (avsc_satd_4x4_avx2, u8),
    (avsc_satd_4x8_avx2, u8),
    (avsc_satd_4x16_avx2, u8),
    (avsc_satd_8x4_avx2, u8),
    (avsc_satd_8x8_avx2, u8),
    (avsc_satd_8x16_avx2, u8),
    (avsc_satd_8x32_avx2, u8),
    (avsc_satd_16x4_avx2, u8),
    (avsc_satd_16x8_avx2, u8),
    (avsc_satd_16x16_avx2, u8),
    (avsc_satd_16x32_avx2, u8),
    (avsc_satd_16x64_avx2, u8),
    (avsc_satd_32x8_avx2, u8),
    (avsc_satd_32x16_avx2, u8),
    (avsc_satd_32x32_avx2, u8),
    (avsc_satd_32x64_avx2, u8),
    (avsc_satd_64x16_avx2, u8),
    (avsc_satd_64x32_avx2, u8),
    (avsc_satd_64x64_avx2, u8),
    (avsc_satd_64x128_avx2, u8),
    (avsc_satd_128x64_avx2, u8),
    (avsc_satd_128x128_avx2, u8),
    // SATD HBD
    (avsc_satd_4x4_hbd_avx2, u16),
    (avsc_satd_4x8_hbd_avx2, u16),
    (avsc_satd_4x16_hbd_avx2, u16),
    (avsc_satd_8x4_hbd_avx2, u16),
    (avsc_satd_8x8_hbd_avx2, u16),
    (avsc_satd_8x16_hbd_avx2, u16),
    (avsc_satd_8x32_hbd_avx2, u16),
    (avsc_satd_16x4_hbd_avx2, u16),
    (avsc_satd_16x8_hbd_avx2, u16),
    (avsc_satd_16x16_hbd_avx2, u16),
    (avsc_satd_16x32_hbd_avx2, u16),
    (avsc_satd_16x64_hbd_avx2, u16),
    (avsc_satd_32x8_hbd_avx2, u16),
    (avsc_satd_32x16_hbd_avx2, u16),
    (avsc_satd_32x32_hbd_avx2, u16),
    (avsc_satd_32x64_hbd_avx2, u16),
    (avsc_satd_64x16_hbd_avx2, u16),
    (avsc_satd_64x32_hbd_avx2, u16),
    (avsc_satd_64x64_hbd_avx2, u16),
    (avsc_satd_64x128_hbd_avx2, u16),
    (avsc_satd_128x64_hbd_avx2, u16),
    (avsc_satd_128x128_hbd_avx2, u16)
];

#[target_feature(enable = "avx2")]
pub(super) fn get_satd_internal<T: Pixel>(
    src: &PlaneRegion<'_, T>,
    dst: &PlaneRegion<'_, T>,
    w: usize,
    h: usize,
    bit_depth: usize,
) -> u32 {
    let bsize_opt = BlockSize::from_width_and_height_opt(w, h);

    match (bsize_opt, T::type_enum()) {
        (Err(_), _) => super::rust::get_satd_internal(src, dst, w, h, bit_depth),
        (Ok(bsize), PixelType::U8) => unsafe {
            (match bsize {
                BlockSize::BLOCK_4X4 => avsc_satd_4x4_avx2,
                BlockSize::BLOCK_4X8 => avsc_satd_4x8_avx2,
                BlockSize::BLOCK_4X16 => avsc_satd_4x16_avx2,
                BlockSize::BLOCK_8X4 => avsc_satd_8x4_avx2,
                BlockSize::BLOCK_16X4 => avsc_satd_16x4_avx2,
                BlockSize::BLOCK_8X8 => avsc_satd_8x8_avx2,
                BlockSize::BLOCK_8X16 => avsc_satd_8x16_avx2,
                BlockSize::BLOCK_8X32 => avsc_satd_8x32_avx2,
                BlockSize::BLOCK_16X8 => avsc_satd_16x8_avx2,
                BlockSize::BLOCK_16X16 => avsc_satd_16x16_avx2,
                BlockSize::BLOCK_16X32 => avsc_satd_16x32_avx2,
                BlockSize::BLOCK_16X64 => avsc_satd_16x64_avx2,
                BlockSize::BLOCK_32X8 => avsc_satd_32x8_avx2,
                BlockSize::BLOCK_32X16 => avsc_satd_32x16_avx2,
                BlockSize::BLOCK_32X32 => avsc_satd_32x32_avx2,
                BlockSize::BLOCK_32X64 => avsc_satd_32x64_avx2,
                BlockSize::BLOCK_64X16 => avsc_satd_64x16_avx2,
                BlockSize::BLOCK_64X32 => avsc_satd_64x32_avx2,
                BlockSize::BLOCK_64X64 => avsc_satd_64x64_avx2,
                BlockSize::BLOCK_64X128 => avsc_satd_64x128_avx2,
                BlockSize::BLOCK_128X64 => avsc_satd_128x64_avx2,
                BlockSize::BLOCK_128X128 => avsc_satd_128x128_avx2,
            })(
                src.data_ptr() as *const _,
                T::to_asm_stride(src.plane_cfg.stride),
                dst.data_ptr() as *const _,
                T::to_asm_stride(dst.plane_cfg.stride),
            )
        },
        (Ok(bsize), PixelType::U16) => unsafe {
            (match bsize {
                BlockSize::BLOCK_4X4 => avsc_satd_4x4_hbd_avx2,
                BlockSize::BLOCK_4X8 => avsc_satd_4x8_hbd_avx2,
                BlockSize::BLOCK_4X16 => avsc_satd_4x16_hbd_avx2,
                BlockSize::BLOCK_8X4 => avsc_satd_8x4_hbd_avx2,
                BlockSize::BLOCK_16X4 => avsc_satd_16x4_hbd_avx2,
                BlockSize::BLOCK_8X8 => avsc_satd_8x8_hbd_avx2,
                BlockSize::BLOCK_8X16 => avsc_satd_8x16_hbd_avx2,
                BlockSize::BLOCK_8X32 => avsc_satd_8x32_hbd_avx2,
                BlockSize::BLOCK_16X8 => avsc_satd_16x8_hbd_avx2,
                BlockSize::BLOCK_16X16 => avsc_satd_16x16_hbd_avx2,
                BlockSize::BLOCK_16X32 => avsc_satd_16x32_hbd_avx2,
                BlockSize::BLOCK_16X64 => avsc_satd_16x64_hbd_avx2,
                BlockSize::BLOCK_32X8 => avsc_satd_32x8_hbd_avx2,
                BlockSize::BLOCK_32X16 => avsc_satd_32x16_hbd_avx2,
                BlockSize::BLOCK_32X32 => avsc_satd_32x32_hbd_avx2,
                BlockSize::BLOCK_32X64 => avsc_satd_32x64_hbd_avx2,
                BlockSize::BLOCK_64X16 => avsc_satd_64x16_hbd_avx2,
                BlockSize::BLOCK_64X32 => avsc_satd_64x32_hbd_avx2,
                BlockSize::BLOCK_64X64 => avsc_satd_64x64_hbd_avx2,
                BlockSize::BLOCK_64X128 => avsc_satd_64x128_hbd_avx2,
                BlockSize::BLOCK_128X64 => avsc_satd_128x64_hbd_avx2,
                BlockSize::BLOCK_128X128 => avsc_satd_128x128_hbd_avx2,
            })(
                src.data_ptr() as *const _,
                T::to_asm_stride(src.plane_cfg.stride),
                dst.data_ptr() as *const _,
                T::to_asm_stride(dst.plane_cfg.stride),
            )
        },
    }
}
