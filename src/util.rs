// Copyright (c) 2017-2019, The rav1e contributors. All rights reserved
//
// This source code is subject to the terms of the BSD 2 Clause License and
// the Alliance for Open Media Patent License 1.0. If the BSD 2 Clause License
// was not distributed with this source code in the LICENSE file, you can
// obtain it at www.aomedia.org/license/software. If the Alliance for Open
// Media Patent License 1.0 was not distributed with this source code in the
// PATENTS file, you can obtain it at www.aomedia.org/license/patent.

//! Traits for generic code over low and high bit depth video.
//!
//! Borrowed from rav1e.

use std::fmt::Debug;
use std::mem::MaybeUninit;

/// Sample position for subsampled chroma
#[derive(Copy, Clone, Debug, PartialEq)]
pub(crate) enum ChromaSamplePosition {
    /// The source video transfer function must be signaled
    /// outside the AV1 bitstream.
    Unknown,
    /// Horizontally co-located with (0, 0) luma sample, vertically positioned
    /// in the middle between two luma samples.
    Vertical,
    /// Co-located with (0, 0) luma sample.
    Colocated,
}

impl Default for ChromaSamplePosition {
    fn default() -> Self {
        ChromaSamplePosition::Unknown
    }
}

// TODO: use the num crate?
/// A rational number.
#[derive(Clone, Copy, Debug)]
pub struct Rational {
    /// Numerator.
    pub num: u64,
    /// Denominator.
    pub den: u64,
}

impl Rational {
    /// Creates a rational number from the given numerator and denominator.
    pub const fn new(num: u64, den: u64) -> Self {
        Rational { num, den }
    }
}

#[repr(align(32))]
pub struct Align32;

// A 16 byte aligned array.
// # Examples
// ```
// let mut x: AlignedArray<[i16; 64 * 64]> = AlignedArray::new([0; 64 * 64]);
// assert!(x.array.as_ptr() as usize % 16 == 0);
//
// let mut x: AlignedArray<[i16; 64 * 64]> = AlignedArray::uninitialized();
// assert!(x.array.as_ptr() as usize % 16 == 0);
// ```
pub struct AlignedArray<ARRAY> {
    _alignment: [Align32; 0],
    pub array: ARRAY,
}

impl<A> AlignedArray<A> {
    pub const fn new(array: A) -> Self {
        AlignedArray {
            _alignment: [],
            array,
        }
    }
    #[allow(clippy::uninit_assumed_init)]
    pub fn uninitialized() -> Self {
        Self::new(unsafe { MaybeUninit::uninit().assume_init() })
    }
}

#[test]
fn sanity() {
    fn is_aligned<T>(ptr: *const T, n: usize) -> bool {
        ((ptr as usize) & ((1 << n) - 1)) == 0
    }

    let a: AlignedArray<_> = AlignedArray::new([0u8; 3]);
    assert!(is_aligned(a.array.as_ptr(), 4));
}
