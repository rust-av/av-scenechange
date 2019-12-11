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

use num_traits::{AsPrimitive, PrimInt};
use std::fmt::{Debug, Display};
use std::mem::{size_of, MaybeUninit};
use y4m::Colorspace;

/// Defines a type which supports being cast to from a generic integer type.
///
/// Intended for casting to and from a [`Pixel`](trait.Pixel.html).
pub trait CastFromPrimitive<T>: Copy + 'static {
    /// Cast from a generic integer type to the given type.
    fn cast_from(v: T) -> Self;
}

macro_rules! impl_cast_from_primitive {
  ( $T:ty => $U:ty ) => {
    impl CastFromPrimitive<$U> for $T {
      #[inline(always)]
      fn cast_from(v: $U) -> Self { v as Self }
    }
  };
  ( $T:ty => { $( $U:ty ),* } ) => {
    $( impl_cast_from_primitive!($T => $U); )*
  };
}

// casts to { u8, u16 } are implemented separately using Pixel, so that the
// compiler understands that CastFromPrimitive<T: Pixel> is always implemented
impl_cast_from_primitive!(u8 => { u32, u64, usize });
impl_cast_from_primitive!(u8 => { i8, i16, i32, i64, isize });
impl_cast_from_primitive!(u16 => { u32, u64, usize });
impl_cast_from_primitive!(u16 => { i8, i16, i32, i64, isize });
impl_cast_from_primitive!(i16 => { u32, u64, usize });
impl_cast_from_primitive!(i16 => { i8, i16, i32, i64, isize });
impl_cast_from_primitive!(i32 => { u32, u64, usize });
impl_cast_from_primitive!(i32 => { i8, i16, i32, i64, isize });

#[doc(hidden)]
pub enum PixelType {
    U8,
    U16,
}

/// A trait for types which may represent a pixel in a video.
/// Currently implemented for `u8` and `u16`.
/// `u8` should be used for low-bit-depth video, and `u16`
/// for high-bit-depth video.
pub trait Pixel:
    PrimInt
    + Into<u32>
    + Into<i32>
    + AsPrimitive<u8>
    + AsPrimitive<i16>
    + AsPrimitive<u16>
    + AsPrimitive<i32>
    + AsPrimitive<u32>
    + AsPrimitive<usize>
    + CastFromPrimitive<u8>
    + CastFromPrimitive<i16>
    + CastFromPrimitive<u16>
    + CastFromPrimitive<i32>
    + CastFromPrimitive<u32>
    + CastFromPrimitive<usize>
    + Debug
    + Display
    + Send
    + Sync
    + 'static
{
    #[doc(hidden)]
    fn type_enum() -> PixelType;

    #[doc(hidden)]
    /// Converts stride in pixels to stride in bytes.
    fn to_asm_stride(in_stride: usize) -> isize {
        (in_stride * size_of::<Self>()) as isize
    }
}

impl Pixel for u8 {
    fn type_enum() -> PixelType {
        PixelType::U8
    }
}

impl Pixel for u16 {
    fn type_enum() -> PixelType {
        PixelType::U16
    }
}

macro_rules! impl_cast_from_pixel_to_primitive {
    ( $T:ty ) => {
        impl<T: Pixel> CastFromPrimitive<T> for $T {
            #[inline(always)]
            fn cast_from(v: T) -> Self {
                v.as_()
            }
        }
    };
}

impl_cast_from_pixel_to_primitive!(u8);
impl_cast_from_pixel_to_primitive!(i16);
impl_cast_from_pixel_to_primitive!(u16);
impl_cast_from_pixel_to_primitive!(i32);
impl_cast_from_pixel_to_primitive!(u32);

/// Available chroma sampling formats.
#[derive(Copy, Clone, Debug, PartialEq)]
pub enum ChromaSampling {
    /// Both vertically and horizontally subsampled.
    Cs420,
    /// Horizontally subsampled.
    Cs422,
    /// Not subsampled.
    Cs444,
    /// Monochrome.
    Cs400,
}

impl From<Colorspace> for ChromaSampling {
    fn from(other: Colorspace) -> Self {
        use Colorspace::*;
        match other {
            Cmono => ChromaSampling::Cs400,
            C420 | C420p10 | C420p12 | C420jpeg | C420paldv | C420mpeg2 => ChromaSampling::Cs420,
            C422 | C422p10 | C422p12 => ChromaSampling::Cs422,
            C444 | C444p10 | C444p12 => ChromaSampling::Cs444,
        }
    }
}

impl ChromaSampling {
    /// Provides the amount to right shift the luma plane dimensions to get the
    ///  chroma plane dimensions.
    /// Only values 0 or 1 are ever returned.
    /// The plane dimensions must also be rounded up to accommodate odd luma plane
    ///  sizes.
    /// Cs400 returns None, as there are no chroma planes.
    pub fn get_decimation(self) -> Option<(usize, usize)> {
        use self::ChromaSampling::*;
        match self {
            Cs420 => Some((1, 1)),
            Cs422 => Some((1, 0)),
            Cs444 => Some((0, 0)),
            Cs400 => None,
        }
    }

    /// Calculates the size of a chroma plane for this sampling type, given the luma plane dimensions.
    pub fn get_chroma_dimensions(self, luma_width: usize, luma_height: usize) -> (usize, usize) {
        if let Some((ss_x, ss_y)) = self.get_decimation() {
            ((luma_width + ss_x) >> ss_x, (luma_height + ss_y) >> ss_y)
        } else {
            (0, 0)
        }
    }
}

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

pub trait Fixed {
    fn floor_log2(&self, n: usize) -> usize;
    fn ceil_log2(&self, n: usize) -> usize;
    fn align_power_of_two(&self, n: usize) -> usize;
    fn align_power_of_two_and_shift(&self, n: usize) -> usize;
}

impl Fixed for usize {
    #[inline(always)]
    fn floor_log2(&self, n: usize) -> usize {
        self & !((1 << n) - 1)
    }

    #[inline(always)]
    fn ceil_log2(&self, n: usize) -> usize {
        (self + (1 << n) - 1).floor_log2(n)
    }

    #[inline(always)]
    fn align_power_of_two(&self, n: usize) -> usize {
        self.ceil_log2(n)
    }

    #[inline(always)]
    fn align_power_of_two_and_shift(&self, n: usize) -> usize {
        (self + (1 << n) - 1) >> n
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

#[inline(always)]
pub fn msb(x: i32) -> i32 {
    debug_assert!(x > 0);
    31 ^ (x.leading_zeros() as i32)
}

#[inline(always)]
pub const fn round_shift(value: i32, bit: usize) -> i32 {
    (value + (1 << bit >> 1)) >> bit
}
