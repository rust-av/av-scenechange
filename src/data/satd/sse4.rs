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
    (avsc_satd_4x4_sse4, u8)
];

#[target_feature(enable = "sse4.1")]
pub(super) fn get_satd_internal<T: Pixel>(
    src: &PlaneRegion<'_, T>,
    dst: &PlaneRegion<'_, T>,
    w: usize,
    h: usize,
    bit_depth: usize,
) -> u32 {
    let bsize_opt = BlockSize::from_width_and_height_opt(w, h);

    match (bsize_opt, size_of::<T>()) {
        // SAFETY: call to SIMD function
        (Ok(bsize), 1) => unsafe {
            match bsize {
                BlockSize::BLOCK_4X4 => avsc_satd_4x4_sse4(
                    src.data_ptr() as *const _,
                    (size_of::<T>() * src.plane_cfg.stride.get()) as isize,
                    dst.data_ptr() as *const _,
                    (size_of::<T>() * dst.plane_cfg.stride.get()) as isize,
                ),
                _ => super::ssse3::get_satd_internal(src, dst, w, h, bit_depth),
            }
        },
        _ => super::ssse3::get_satd_internal(src, dst, w, h, bit_depth),
    }
}
