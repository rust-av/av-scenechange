use v_frame::{
    pixel::{CastFromPrimitive, Pixel},
    plane::Plane,
};

use crate::data::plane::{Area, PlaneRegion, Rect};

pub(super) fn sad_plane_internal<T: Pixel>(src: &Plane<T>, dst: &Plane<T>) -> u64 {
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

pub(super) fn get_sad_internal<T: Pixel>(
    plane_org: &PlaneRegion<'_, T>,
    plane_ref: &PlaneRegion<'_, T>,
    w: usize,
    h: usize,
    _bit_depth: usize,
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
