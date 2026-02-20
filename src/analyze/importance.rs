#[cfg(test)]
mod tests;

use std::sync::Arc;

use cfg_if::cfg_if;
use v_frame::{frame::Frame, pixel::Pixel, plane::Plane};

use super::intra::BLOCK_TO_PLANE_SHIFT;

/// Size of blocks for the importance computation, in pixels.
pub const IMPORTANCE_BLOCK_SIZE: usize =
    1 << (IMPORTANCE_BLOCK_TO_BLOCK_SHIFT + BLOCK_TO_PLANE_SHIFT);
pub const IMPORTANCE_BLOCK_TO_BLOCK_SHIFT: usize = 1;
pub const IMP_BLOCK_MV_UNITS_PER_PIXEL: i64 = 8;
pub const IMP_BLOCK_SIZE_IN_MV_UNITS: i64 =
    IMPORTANCE_BLOCK_SIZE as i64 * IMP_BLOCK_MV_UNITS_PER_PIXEL;

pub(crate) fn estimate_importance_block_difference<T: Pixel>(
    frame: &Arc<Frame<T>>,
    ref_frame: &Arc<Frame<T>>,
) -> f64 {
    let plane_org = &frame.y_plane;
    let plane_ref = &ref_frame.y_plane;
    let h_in_imp_b = plane_org.height().get() / IMPORTANCE_BLOCK_SIZE;
    let w_in_imp_b = plane_org.width().get() / IMPORTANCE_BLOCK_SIZE;

    let mut imp_block_costs = 0;

    (0..h_in_imp_b).for_each(|y| {
        (0..w_in_imp_b).for_each(|x| {
            let histogram_org_sum = sum_8x8_block(plane_org, x, y);
            let histogram_ref_sum = sum_8x8_block(plane_ref, x, y);

            let count = (IMPORTANCE_BLOCK_SIZE * IMPORTANCE_BLOCK_SIZE) as i64;

            let mean = (((histogram_org_sum + count / 2) / count)
                - ((histogram_ref_sum + count / 2) / count))
                .abs();

            imp_block_costs += mean as u64;
        });
    });

    imp_block_costs as f64 / (w_in_imp_b * h_in_imp_b) as f64
}

fn sum_8x8_block<T: Pixel>(plane: &Plane<T>, x: usize, y: usize) -> i64 {
    cfg_if! {
        if #[cfg(asm_x86_64)] {
            // SAFETY: SSE2 is baseline on all x86_64 CPUs.
            // Pointer accesses stay within the plane's allocated data buffer,
            // as the caller guarantees valid block coordinates.
            unsafe { sum_8x8_block_sse2(plane, x, y) }
        } else {
            sum_8x8_block_rust(plane, x, y)
        }
    }
}

#[cfg(any(not(asm_x86_64), test))]
fn sum_8x8_block_rust<T: Pixel>(plane: &Plane<T>, x: usize, y: usize) -> i64 {
    use crate::data::get_dbg;

    // Coordinates of the top-left corner of the reference block, in MV units.
    let x = x * IMPORTANCE_BLOCK_SIZE;
    let y = y * IMPORTANCE_BLOCK_SIZE;

    let data = get_dbg(plane.data(), plane.data_origin()..);
    let stride = plane.geometry().stride.get();
    (y..(y + 8)).fold(0, |acc, row_idx| {
        let row = get_dbg(
            get_dbg(data, (x + row_idx * stride)..),
            ..IMPORTANCE_BLOCK_SIZE,
        );
        // 16-bit precision is sufficient for an 8 px row,
        // as `IMPORTANCE_BLOCK_SIZE * (2^12 - 1) < 2^16 - 1`,
        // so overflow is not possible
        acc + row
            .iter()
            .map(|pixel| pixel.to_u16().expect("value should fit in u16"))
            .sum::<u16>() as i64
    })
}

#[cfg(asm_x86_64)]
#[target_feature(enable = "sse2")]
unsafe fn sum_8x8_block_sse2<T: Pixel>(plane: &Plane<T>, x: usize, y: usize) -> i64 {
    use std::arch::x86_64::*;

    let bx = x * IMPORTANCE_BLOCK_SIZE;
    let by = y * IMPORTANCE_BLOCK_SIZE;
    let stride = plane.geometry().stride.get();
    let origin = plane.data_origin();

    // SAFETY: All pointer arithmetic below stays within the plane's allocated
    // data buffer. The caller guarantees valid block coordinates (x, y), and
    // the plane's data extends at least `(by + 7) * stride + bx + 8` elements
    // past `data_origin()`. All SSE2 intrinsics used are baseline x86_64.
    unsafe {
        match size_of::<T>() {
            1 => {
                // u8 path: _mm_sad_epu8 with zero sums 8 bytes in one instruction
                let base_ptr = plane.data().as_ptr().add(origin).cast::<u8>();
                let zero = _mm_setzero_si128();
                let mut acc = _mm_setzero_si128();

                for row in 0..IMPORTANCE_BLOCK_SIZE {
                    let row_ptr = base_ptr.add(bx + (by + row) * stride);
                    let pixels = _mm_loadl_epi64(row_ptr.cast());
                    acc = _mm_add_epi64(acc, _mm_sad_epu8(pixels, zero));
                }

                // acc has the total sum in the low 64-bit lane
                // (upper lane is zero since we only loaded 8 bytes via loadl)
                _mm_cvtsi128_si64(acc)
            }
            2 => {
                // u16 path: 8 x u16 = 128 bits = exactly one XMM register per row
                let base_ptr = plane.data().as_ptr().add(origin).cast::<u16>();
                let ones = _mm_set1_epi16(1);
                let mut acc = _mm_setzero_si128();

                for row in 0..IMPORTANCE_BLOCK_SIZE {
                    let row_ptr = base_ptr.add(bx + (by + row) * stride);
                    let pixels = _mm_loadu_si128(row_ptr.cast());
                    // Safe to accumulate in u16: 8 rows * max 4095 = 32760 < 65535
                    acc = _mm_add_epi16(acc, pixels);
                }

                // Horizontal sum: _mm_madd_epi16 with ones pairs adjacent u16 → 4 x i32
                // (safe since max 32760 < i16::MAX = 32767)
                let paired = _mm_madd_epi16(acc, ones);
                let hi64 = _mm_srli_si128::<8>(paired);
                let sum2 = _mm_add_epi32(paired, hi64);
                let hi32 = _mm_srli_si128::<4>(sum2);
                let total = _mm_add_epi32(sum2, hi32);

                _mm_cvtsi128_si32(total) as i64
            }
            _ => unreachable!(),
        }
    }
}
