use v_frame::{
    pixel::{Pixel, PixelType},
    plane::PlaneSlice,
};

use crate::data::plane::PlaneRegionMut;

#[allow(clippy::too_many_arguments)]
#[target_feature(enable = "avx2")]
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

        match T::type_enum() {
            PixelType::U8 => avsc_put_8tap_regular_8bpc_avx2(
                dst.data_ptr_mut() as *mut _,
                T::to_asm_stride(dst.plane_cfg.stride),
                src.as_ptr() as *const _,
                T::to_asm_stride(src.plane.cfg.stride),
                width as i32,
                height as i32,
                col_frac,
                row_frac,
            ),
            PixelType::U16 => avsc_put_8tap_regular_16bpc_avx2(
                dst.data_ptr_mut() as *mut _,
                T::to_asm_stride(dst.plane_cfg.stride),
                src.as_ptr() as *const _,
                T::to_asm_stride(src.plane.cfg.stride),
                width as i32,
                height as i32,
                col_frac,
                row_frac,
            ),
        }
    }
}

unsafe extern "C" {
    unsafe fn avsc_put_8tap_regular_8bpc_avx2(
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
    unsafe fn avsc_put_8tap_regular_16bpc_avx2(
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
