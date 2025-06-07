use v_frame::{
    pixel::{Pixel, PixelType},
    plane::Plane,
};

extern "C" {
    fn avsc_sad_plane_8bpc_sse2(
        src: *const u8,
        dst: *const u8,
        stride: libc::size_t,
        width: libc::size_t,
        rows: libc::size_t,
    ) -> u64;
}

#[target_feature(enable = "sse2")]
pub(super) fn sad_plane_internal<T: Pixel>(src: &Plane<T>, dst: &Plane<T>) -> u64 {
    use std::mem;

    assert_eq!(src.cfg.width, dst.cfg.width);
    assert_eq!(src.cfg.stride, dst.cfg.stride);
    assert_eq!(src.cfg.height, dst.cfg.height);
    assert!(src.cfg.width <= src.cfg.stride);

    match T::type_enum() {
        PixelType::U8 => unsafe {
            avsc_sad_plane_8bpc_sse2(
                mem::transmute::<*const T, *const u8>(src.data_origin().as_ptr()),
                mem::transmute::<*const T, *const u8>(dst.data_origin().as_ptr()),
                src.cfg.stride,
                src.cfg.width,
                src.cfg.height,
            )
        },
        PixelType::U16 => super::rust::sad_plane_internal(src, dst),
    }
}
