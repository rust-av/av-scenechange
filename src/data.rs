use std::mem::MaybeUninit;

use semisafe::result::unwrap as res_unwrap;
use v_frame::pixel::Pixel;

pub(crate) mod block;
pub(crate) mod frame;
pub(crate) mod hadamard;
pub(crate) mod motion;
pub(crate) mod plane;
pub(crate) mod prediction;
pub(crate) mod sad;
pub(crate) mod satd;
pub(crate) mod superblock;
pub(crate) mod tile;

#[cfg(feature = "bench-internals")]
pub use self::motion::FrameMEStats;

/// Assume all the elements are initialized.
pub unsafe fn slice_assume_init_mut<T: Copy>(slice: &'_ mut [MaybeUninit<T>]) -> &'_ mut [T] {
    // SAFETY: caller must assume elements are initialized
    unsafe { &mut *(slice as *mut [MaybeUninit<T>] as *mut [T]) }
}

#[inline]
pub(crate) fn pixel_as_i32<T: Pixel>(pixel: T) -> i32 {
    i32::from(pixel.into())
}

#[inline]
pub(crate) fn pixel_as_u32<T: Pixel>(pixel: T) -> u32 {
    u32::from(pixel.into())
}

#[inline]
pub(crate) fn pixel_from_u16<T: Pixel>(value: u16) -> T {
    res_unwrap(T::try_from(value))
}
