use v_frame::pixel::Pixel;

use crate::data::plane::PlaneRegionMut;

#[allow(clippy::too_many_arguments)]
#[target_feature(enable = "neon")]
pub fn put_8tap_internal<T: Pixel>(
    dst: &mut PlaneRegionMut<'_, T>,
    src: PlaneSlice<'_, T>,
    width: usize,
    height: usize,
    col_frac: i32,
    row_frac: i32,
    _bit_depth: usize,
) {
    // SAFETY: The assembly only supports even heights and valid uncropped
    //         widths
    unsafe {
        assert_eq!(height & 1, 0);
        assert!(width.is_power_of_two() && (2..=128).contains(&width));

        // SAFETY: Check bounds of dst
        assert!(dst.rect().width >= width && dst.rect().height >= height);

        // SAFETY: Check bounds of src
        assert!(src.accessible(width + 4, height + 4));
        assert!(src.accessible_neg(3, 3));

        match size_of::<T>() {
            1 => avsc_put_8tap_regular_8bpc_neon(
                dst.data_ptr_mut() as *mut _,
                (size_of::<T>() * dst.plane_cfg.stride.get()) as isize,
                src.as_ptr() as *const _,
                (size_of::<T>() * src.plane_cfg.stride.get()) as isize,
                width as i32,
                height as i32,
                col_frac,
                row_frac,
            ),
            2 => avsc_put_8tap_regular_16bpc_neon(
                dst.data_ptr_mut() as *mut _,
                (size_of::<T>() * dst.plane_cfg.stride.get()) as isize,
                src.as_ptr() as *const _,
                (size_of::<T>() * src.plane_cfg.stride.get()) as isize,
                width as i32,
                height as i32,
                col_frac,
                row_frac,
            ),
            _ => unreachable!(),
        }
    }
}

unsafe extern "C" {
    unsafe fn avsc_put_8tap_regular_8bpc_neon(
        dst: *mut u8,
        dst_stride: isize,
        src: *const u8,
        src_stride: isize,
        w: i32,
        h: i32,
        mx: i32,
        my: i32,
    );
}

unsafe extern "C" {
    unsafe fn avsc_put_8tap_regular_16bpc_neon(
        dst: *mut u16,
        dst_stride: isize,
        src: *const u16,
        src_stride: isize,
        w: i32,
        h: i32,
        mx: i32,
        my: i32,
    );
}
