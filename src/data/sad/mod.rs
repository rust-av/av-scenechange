#[cfg(asm_x86_64)]
mod avx2;
mod rust;
#[cfg(asm_x86_64)]
mod sse2;

#[cfg(test)]
mod tests;

use cfg_if::cfg_if;
use v_frame::{pixel::Pixel, plane::Plane};

use super::plane::PlaneRegion;

/// Compute the sum of absolute differences (SADs) on 2 rows of pixels
///
/// This differs from other SAD functions in that it operates over a row
/// (or line) of unknown length rather than a `PlaneRegion<T>`.
pub(crate) fn sad_plane<T: Pixel>(src: &Plane<T>, dst: &Plane<T>) -> u64 {
    cfg_if! {
        if #[cfg(asm_x86_64)] {
            if crate::cpu::has_avx2() {
                unsafe { avx2::sad_plane_internal(src, dst) }
            } else {
                // All x86_64 CPUs have SSE2
                unsafe { sse2::sad_plane_internal(src, dst) }
            }
        } else {
            rust::sad_plane_internal(src, dst)
        }
    }
}

pub(crate) fn get_sad<T: Pixel>(
    plane_org: &PlaneRegion<'_, T>,
    plane_ref: &PlaneRegion<'_, T>,
    w: usize,
    h: usize,
    bit_depth: usize,
) -> u32 {
    rust::get_sad_internal(plane_org, plane_ref, w, h, bit_depth)
}
