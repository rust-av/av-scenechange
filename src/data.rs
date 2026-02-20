use std::mem::MaybeUninit;

pub(crate) mod block;
pub(crate) mod frame;
pub(crate) mod hadamard;
pub(crate) mod mc;
pub(crate) mod motion;
pub(crate) mod plane;
pub(crate) mod prediction;
pub(crate) mod sad;
pub(crate) mod satd;
pub(crate) mod superblock;
pub(crate) mod tile;

/// Assume all the elements are initialized.
pub unsafe fn slice_assume_init_mut<T: Copy>(slice: &'_ mut [MaybeUninit<T>]) -> &'_ mut [T] {
    // SAFETY: caller must assume elements are initialized
    unsafe { &mut *(slice as *mut [MaybeUninit<T>] as *mut [T]) }
}

#[allow(
    clippy::inline_always,
    reason = "intended as a thin compile-time-elided wrapper"
)]
#[inline(always)]
#[cfg(any(not(asm_x86_64), test))]
pub fn get_dbg<T, I: std::slice::SliceIndex<[T]>>(
    arr: &[T],
    index: I,
) -> &<I as std::slice::SliceIndex<[T]>>::Output {
    use cfg_if::cfg_if;

    cfg_if! {
        if #[cfg(debug_assertions)] {
            arr.get(index).expect("array index out of bounds")
        } else {
            unsafe{ arr.get_unchecked(index) }
        }
    }
}
