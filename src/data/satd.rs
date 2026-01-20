#[cfg(asm_x86_64)]
mod avx2;
#[cfg(asm_neon)]
mod neon;
mod rust;
#[cfg(asm_x86_64)]
mod sse4;
#[cfg(asm_x86_64)]
mod ssse3;

#[cfg(test)]
mod tests;

use std::num::NonZeroUsize;

use cfg_if::cfg_if;
use v_frame::pixel::Pixel;

use super::plane::PlaneRegion;

pub(crate) fn get_satd<T: Pixel>(
    src: &PlaneRegion<'_, T>,
    dst: &PlaneRegion<'_, T>,
    w: NonZeroUsize,
    h: NonZeroUsize,
    bit_depth: NonZeroUsize,
) -> u32 {
    cfg_if! {
        if #[cfg(asm_x86_64)] {
            if crate::cpu::has_avx2() {
                // SAFETY: call to SIMD function
                unsafe { return avx2::get_satd_internal(src, dst, w, h, bit_depth); }
            } else if crate::cpu::has_sse4() {
                // SAFETY: call to SIMD function
                unsafe { return sse4::get_satd_internal(src, dst, w, h, bit_depth); }
            } else if crate::cpu::has_ssse3() {
                // SAFETY: call to SIMD function
                unsafe { return ssse3::get_satd_internal(src, dst, w, h, bit_depth); }
            }
        } else if #[cfg(asm_neon)] {
            // SAFETY: call to SIMD function
            unsafe { neon::get_satd_internal(src, dst, w, h, bit_depth) }
        }
    }

    #[cfg(not(asm_neon))]
    rust::get_satd_internal(src, dst, w, h, bit_depth)
}
