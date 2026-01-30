use v_frame::{pixel::Pixel, plane::Plane};

use crate::data::plane::{Area, PlaneRegion, Rect};

pub(super) fn sad_plane_internal<T: Pixel>(src: &Plane<T>, dst: &Plane<T>) -> u64 {
    assert_eq!(src.width(), dst.width());
    assert_eq!(src.height(), dst.height());

    src.rows()
        .zip(dst.rows())
        .map(|(src, dst)| {
            src.iter()
                .zip(dst.iter())
                .map(|(&p1, &p2)| {
                    let p1 = p1.to_i32().expect("value should fit in i32");
                    let p2 = p2.to_i32().expect("value should fit in i32");
                    p1.abs_diff(p2)
                })
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
                .map(|(&p1, &p2)| {
                    let p1 = p1.to_i32().expect("value should fit in i32");
                    let p2 = p2.to_i32().expect("value should fit in i32");
                    p1.abs_diff(p2)
                })
                .sum::<u32>()
        })
        .sum()
}
