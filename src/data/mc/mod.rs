#[cfg(asm_x86_64)]
mod avx2;
#[cfg(asm_x86_64)]
mod avx512icl;
#[cfg(asm_neon)]
mod neon;
#[cfg(not(asm_neon))]
mod rust;
#[cfg(asm_x86_64)]
mod ssse3;

use cfg_if::cfg_if;
use v_frame::{pixel::Pixel, plane::PlaneSlice};

use crate::data::plane::PlaneRegionMut;

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

#[allow(clippy::too_many_arguments)]
pub fn put_8tap<T: Pixel>(
    dst: &mut PlaneRegionMut<'_, T>,
    src: PlaneSlice<'_, T>,
    width: usize,
    height: usize,
    col_frac: i32,
    row_frac: i32,
    bit_depth: usize,
) {
    cfg_if! {
        if #[cfg(asm_x86_64)] {
            if crate::cpu::has_avx512icl() {
                unsafe { avx512icl::put_8tap_internal(dst, src, width, height, col_frac, row_frac, bit_depth); }
                return;
            } else if crate::cpu::has_avx2() {
                unsafe { avx2::put_8tap_internal(dst, src, width, height, col_frac, row_frac, bit_depth); }
                return;
            } else if crate::cpu::has_ssse3() {
                unsafe { ssse3::put_8tap_internal(dst, src, width, height, col_frac, row_frac, bit_depth); }
                return;
            }
        } else if #[cfg(asm_neon)] {
            unsafe { neon::put_8tap_internal(dst, src, width, height, col_frac, row_frac, bit_depth); }
        }
    }

    #[cfg(not(asm_neon))]
    rust::put_8tap_internal(dst, src, width, height, col_frac, row_frac, bit_depth);
}
