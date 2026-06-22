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
    let bsize = BlockSize::from_width_and_height(w, h);

    match size_of::<T>() {
        // SAFETY: call to SIMD function
        1 => unsafe {
            match bsize {
                BlockSize::BLOCK_8X8 => avsc_satd_8x8_ssse3(
                    src.data_ptr() as *const _,
                    (size_of::<T>() * src.plane_cfg.stride()) as isize,
                    dst.data_ptr() as *const _,
                    (size_of::<T>() * dst.plane_cfg.stride()) as isize,
                ),
                _ => super::rust::get_satd_internal(src, dst, w, h, bit_depth),
            }
        },
        _ => super::rust::get_satd_internal(src, dst, w, h, bit_depth),
    }
}
