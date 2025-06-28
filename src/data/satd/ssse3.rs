use v_frame::pixel::{Pixel, PixelType};

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
    (avsc_satd_8x8_ssse3, u8)
];

#[target_feature(enable = "ssse3")]
pub(super) fn get_satd_internal<T: Pixel>(
    src: &PlaneRegion<'_, T>,
    dst: &PlaneRegion<'_, T>,
    w: usize,
    h: usize,
    bit_depth: usize,
) -> u32 {
    let bsize_opt = BlockSize::from_width_and_height_opt(w, h);

    match (bsize_opt, T::type_enum()) {
        (Ok(bsize), PixelType::U8) => unsafe {
            match bsize {
                BlockSize::BLOCK_8X8 => avsc_satd_8x8_ssse3(
                    src.data_ptr() as *const _,
                    T::to_asm_stride(src.plane_cfg.stride),
                    dst.data_ptr() as *const _,
                    T::to_asm_stride(dst.plane_cfg.stride),
                ),
                _ => super::rust::get_satd_internal(src, dst, w, h, bit_depth),
            }
        },
        _ => super::rust::get_satd_internal(src, dst, w, h, bit_depth),
    }
}
