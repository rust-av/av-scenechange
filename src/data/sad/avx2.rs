use v_frame::{pixel::Pixel, plane::Plane};

unsafe extern "C" {
    fn avsc_sad_plane_8bpc_avx2(
        src: *const u8,
        dst: *const u8,
        stride: libc::size_t,
        width: libc::size_t,
        rows: libc::size_t,
    ) -> u64;
}

#[target_feature(enable = "avx2")]
pub(super) fn sad_plane_internal<T: Pixel>(src: &Plane<T>, dst: &Plane<T>) -> u64 {
    assert_eq!(src.geometry().width, dst.geometry().width);
    assert_eq!(src.geometry().stride, dst.geometry().stride);
    assert_eq!(src.geometry().height, dst.geometry().height);
    assert!(src.geometry().width <= src.geometry().stride);

    match size_of::<T>() {
        // SAFETY: call to SIMD function
        1 => unsafe {
            avsc_sad_plane_8bpc_avx2(
                src.data().as_ptr().add(src.data_origin()).cast::<u8>(),
                dst.data().as_ptr().add(dst.data_origin()).cast::<u8>(),
                (size_of::<T>() * src.geometry().stride.get()) as libc::size_t,
                src.width().get(),
                src.height().get(),
            )
        },
        2 => super::sse2::sad_plane_internal(src, dst),
        _ => unreachable!(),
    }
}
