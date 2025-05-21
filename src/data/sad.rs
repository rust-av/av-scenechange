use rust::get_sad_internal;
#[cfg(not(asm_x86_64))]
use rust::*;
#[cfg(asm_x86_64)]
use simd_x86::*;
use v_frame::{pixel::Pixel, plane::Plane};

use super::plane::PlaneRegion;
use crate::cpu::CpuFeatureLevel;

mod rust {
    use v_frame::{
        pixel::{CastFromPrimitive, Pixel},
        plane::Plane,
    };

    use crate::{
        data::plane::{Area, PlaneRegion, Rect},
        CpuFeatureLevel,
    };

    pub(super) fn sad_plane_internal<T: Pixel>(
        src: &Plane<T>,
        dst: &Plane<T>,
        _cpu: CpuFeatureLevel,
    ) -> u64 {
        assert_eq!(src.cfg.width, dst.cfg.width);
        assert_eq!(src.cfg.height, dst.cfg.height);

        src.rows_iter()
            .zip(dst.rows_iter())
            .map(|(src, dst)| {
                src.iter()
                    .zip(dst.iter())
                    .map(|(&p1, &p2)| i32::cast_from(p1).abs_diff(i32::cast_from(p2)))
                    .sum::<u32>() as u64
            })
            .sum()
    }

    pub fn get_sad_internal<T: Pixel>(
        plane_org: &PlaneRegion<'_, T>,
        plane_ref: &PlaneRegion<'_, T>,
        w: usize,
        h: usize,
        _bit_depth: usize,
        _cpu: CpuFeatureLevel,
    ) -> u32 {
        debug_assert!(w <= 128 && h <= 128);
        let plane_org = plane_org.subregion(Area::Rect(Rect {
            x: 0,
            y: 0,
            width: w,
            height: h,
        }));
        let plane_ref = plane_ref.subregion(Area::Rect(Rect {
            x: 0,
            y: 0,
            width: w,
            height: h,
        }));

        plane_org
            .rows_iter()
            .zip(plane_ref.rows_iter())
            .map(|(src, dst)| {
                src.iter()
                    .zip(dst)
                    .map(|(&p1, &p2)| i32::cast_from(p1).abs_diff(i32::cast_from(p2)))
                    .sum::<u32>()
            })
            .sum()
    }
}

#[cfg(asm_x86_64)]
mod simd_x86 {
    use v_frame::{
        pixel::{Pixel, PixelType},
        plane::Plane,
    };

    use crate::CpuFeatureLevel;

    macro_rules! decl_sad_plane_fn {
      ($($f:ident),+) => {
        extern "C" {
          $(
            fn $f(
              src: *const u8, dst: *const u8, stride: libc::size_t,
              width: libc::size_t, rows: libc::size_t
            ) -> u64;
          )*
        }
      };
    }

    decl_sad_plane_fn!(avsc_sad_plane_8bpc_sse2, avsc_sad_plane_8bpc_avx2);

    pub(super) fn sad_plane_internal<T: Pixel>(
        src: &Plane<T>,
        dst: &Plane<T>,
        cpu: CpuFeatureLevel,
    ) -> u64 {
        use std::mem;

        assert_eq!(src.cfg.width, dst.cfg.width);
        assert_eq!(src.cfg.stride, dst.cfg.stride);
        assert_eq!(src.cfg.height, dst.cfg.height);
        assert!(src.cfg.width <= src.cfg.stride);

        match T::type_enum() {
            PixelType::U8 => {
                // helper macro to reduce boilerplate
                macro_rules! call_asm {
                    ($func:ident, $src:expr, $dst:expr, $cpu:expr) => {
                        // SAFETY: Calls Assembly code.
                        unsafe {
                            let result = $func(
                                mem::transmute::<*const T, *const u8>(src.data_origin().as_ptr()),
                                mem::transmute::<*const T, *const u8>(dst.data_origin().as_ptr()),
                                src.cfg.stride,
                                src.cfg.width,
                                src.cfg.height,
                            );

                            result
                        }
                    };
                }

                if cpu >= CpuFeatureLevel::AVX2 {
                    call_asm!(avsc_sad_plane_8bpc_avx2, src, dst, cpu)
                } else if cpu >= CpuFeatureLevel::SSE2 {
                    call_asm!(avsc_sad_plane_8bpc_sse2, src, dst, cpu)
                } else {
                    super::rust::sad_plane_internal(src, dst, cpu)
                }
            }
            PixelType::U16 => super::rust::sad_plane_internal(src, dst, cpu),
        }
    }
}

/// Compute the sum of absolute differences (SADs) on 2 rows of pixels
///
/// This differs from other SAD functions in that it operates over a row
/// (or line) of unknown length rather than a `PlaneRegion<T>`.
pub(crate) fn sad_plane<T: Pixel>(src: &Plane<T>, dst: &Plane<T>, cpu: CpuFeatureLevel) -> u64 {
    sad_plane_internal(src, dst, cpu)
}

pub(crate) fn get_sad<T: Pixel>(
    plane_org: &PlaneRegion<'_, T>,
    plane_ref: &PlaneRegion<'_, T>,
    w: usize,
    h: usize,
    bit_depth: usize,
    cpu: CpuFeatureLevel,
) -> u32 {
    get_sad_internal(plane_org, plane_ref, w, h, bit_depth, cpu)
}
