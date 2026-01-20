use std::{
    iter::FusedIterator,
    marker::PhantomData,
    num::{NonZeroU8, NonZeroUsize},
    ops::{Index, IndexMut, Range},
    slice,
};

use v_frame::{
    frame::FrameBuilder,
    pixel::Pixel,
    plane::{Plane, PlaneGeometry},
};

use super::block::{BLOCK_TO_PLANE_SHIFT, BlockOffset};

/// Absolute offset in pixels inside a plane
#[derive(Clone, Copy, Debug, Default)]
pub struct PlaneOffset {
    pub x: isize,
    pub y: isize,
}

/// Bounded region of a plane
///
/// This allows giving access to a rectangular area of a plane without
/// giving access to the whole plane.
#[derive(Debug)]
pub struct PlaneRegion<'a, T: Pixel> {
    data: *const T, // points to (plane_cfg.x, plane_cfg.y)
    pub plane_cfg: PlaneGeometry,
    // private to guarantee borrowing rules
    rect: Rect,
    phantom: PhantomData<&'a T>,
}

/// Mutable bounded region of a plane
///
/// This allows to give mutable access to a rectangular area of the plane
/// without giving access to the whole plane.
#[derive(Debug)]
pub struct PlaneRegionMut<'a, T: Pixel> {
    data: *mut T, // points to (plane_cfg.x, plane_cfg.y)
    pub plane_cfg: PlaneGeometry,
    pub rect: Rect,
    phantom: PhantomData<&'a mut T>,
}

// common impl for PlaneRegion and PlaneRegionMut
macro_rules! plane_region_common {
  // $name: PlaneRegion or PlaneRegionMut
  // $as_ptr: as_ptr or as_mut_ptr
  // $opt_mut: nothing or mut
  ($name:ident, $as_ptr:ident $(,$opt_mut:tt)?) => {
    impl<'a, T: Pixel> $name<'a, T> {
      /// # Panics
      ///
      /// - If the configured dimensions are invalid
      pub fn from_slice(data: &'a $($opt_mut)? [T], cfg: PlaneGeometry, rect: Rect) -> Self {
        assert!(rect.x >= -(cfg.pad_left as isize));
        assert!(rect.y >= -(cfg.pad_top as isize));
        assert!(cfg.pad_left as isize + rect.x + rect.width.get() as isize <= cfg.stride.get() as isize);
        assert!(cfg.pad_top as isize + rect.y + rect.height.get() as isize <= cfg.alloc_height().get() as isize);

        // SAFETY: The above asserts ensure we do not go OOB.
        unsafe { Self::from_slice_unsafe(data, cfg, rect)}
      }

      unsafe fn from_slice_unsafe(data: &'a $($opt_mut)? [T], cfg: PlaneGeometry, rect: Rect) -> Self {
        let origin = (cfg.pad_top as isize + rect.y) * cfg.stride.get() as isize + cfg.pad_left as isize + rect.x;
        Self {
          // SAFETY: we know that `origin` is within the `data` array
          data: unsafe { data.$as_ptr().offset(origin) },
          plane_cfg: cfg,
          rect,
          phantom: PhantomData,
        }
      }

      #[allow(dead_code)]
      pub fn data_ptr(&self) -> *const T {
        self.data
      }

      pub fn rect(&self) -> &Rect {
        &self.rect
      }

      #[allow(dead_code)]
      pub fn rows_iter(&self) -> PlaneRegionRowsIter<'_, T> {
        PlaneRegionRowsIter {
          data: self.data,
          stride: self.plane_cfg.stride,
          width: self.rect.width,
          remaining: self.rect.height.get(),
          phantom: PhantomData,
        }
      }

      #[allow(dead_code)]
      pub fn vert_windows(&self, h: NonZeroUsize) -> VertWindows<'_, T> {
        VertWindows {
          data: self.data,
          plane_cfg: self.plane_cfg,
          remaining: (self.rect.height.get() as isize - h.get() as isize + 1).max(0) as usize,
          output_rect: Rect {
            x: self.rect.x,
            y: self.rect.y,
            width: self.rect.width,
            height: h
          },
          phantom: PhantomData,
        }
      }

      #[allow(dead_code)]
      pub fn horz_windows(&self, w: NonZeroUsize) -> HorzWindows<'_, T> {
        HorzWindows {
          data: self.data,
          plane_cfg: self.plane_cfg,
          remaining: (self.rect.width.get() as isize - w.get() as isize + 1).max(0) as usize,
          output_rect: Rect {
            x: self.rect.x,
            y: self.rect.y,
            width: w,
            height: self.rect.height
          },
          phantom: PhantomData,
        }
      }

      /// Return a view to a subregion of the plane
      ///
      /// The subregion must be included in (i.e. must not exceed) this region.
      ///
      /// It is described by an `Area`, relative to this region.
      ///
      /// # Panics
      ///
      /// - If the requested dimensions are larger than the plane region size
      ///
      /// # Example
      ///
      /// ``` ignore
      /// # use rav1e::tiling::*;
      /// # fn f(region: &PlaneRegion<'_, u16>) {
      /// // a subregion from (10, 8) to the end of the region
      /// let subregion = region.subregion(Area::StartingAt { x: 10, y: 8 });
      /// # }
      /// ```
      ///
      /// ``` ignore
      /// # use rav1e::context::*;
      /// # use rav1e::tiling::*;
      /// # fn f(region: &PlaneRegion<'_, u16>) {
      /// // a subregion from the top-left of block (2, 3) having size (64, 64)
      /// let bo = BlockOffset { x: 2, y: 3 };
      /// let subregion = region.subregion(Area::BlockRect { bo, width: 64, height: 64 });
      /// # }
      /// ```
      #[allow(dead_code)]
      pub fn subregion(&self, area: Area) -> PlaneRegion<'_, T> {
        if self.data.is_null() {
          unreachable!("DO NOT construct a subregion when data is null");
        }
        let rect = area.to_rect(
          self.plane_cfg.subsampling_x.get() as usize >> 1,
          self.plane_cfg.subsampling_y.get() as usize >> 1,
          self.rect.width,
          self.rect.height,
        );
        assert!(rect.x >= 0 && rect.x as usize <= self.rect.width.get());
        assert!(rect.y >= 0 && rect.y as usize <= self.rect.height.get());
        // SAFETY: The above asserts ensure we do not go outside the original rectangle.
        let data = unsafe {
          self.data.add(rect.y as usize * self.plane_cfg.stride.get() + rect.x as usize)
        };
        let absolute_rect = Rect {
          x: self.rect.x + rect.x,
          y: self.rect.y + rect.y,
          width: rect.width,
          height: rect.height,
        };
        PlaneRegion {
          data,
          plane_cfg: self.plane_cfg,
          rect: absolute_rect,
          phantom: PhantomData,
        }
      }
    }

    // SAFETY: can be safely sent across threads
    unsafe impl<T: Pixel> Send for $name<'_, T> {}
    // SAFETY: this is actually not safe to share across threads...
    // but we do it anyway because the performance impact is large.
    // Callers must be careful not to access overlapping portions of `data`
    unsafe impl<T: Pixel> Sync for $name<'_, T> {}

    impl<T: Pixel> Index<usize> for $name<'_, T> {
      type Output = [T];

      fn index(&self, index: usize) -> &Self::Output {
        assert!(index < self.rect.height.get());
        // SAFETY: The above assert ensures we do not access OOB data.
        unsafe {
          let ptr = self.data.add(index * self.plane_cfg.stride.get());
          slice::from_raw_parts(ptr, self.rect.width.get())
        }
      }
    }
  }
}

plane_region_common!(PlaneRegion, as_ptr);
plane_region_common!(PlaneRegionMut, as_mut_ptr, mut);

impl<'a, T: Pixel> PlaneRegion<'a, T> {
    pub fn new(plane: &'a Plane<T>, rect: Rect) -> Self {
        let geometry = plane.geometry();
        Self::from_slice(plane.data(), geometry, rect)
    }

    pub fn new_from_plane(plane: &'a Plane<T>) -> Self {
        let geometry = plane.geometry();
        let rect = Rect {
            x: 0,
            y: 0,
            width: NonZeroUsize::new(geometry.stride.get() - geometry.pad_left)
                .expect("cannot be zero"),
            height: NonZeroUsize::new(geometry.alloc_height().get() - geometry.pad_top)
                .expect("cannot be zero"),
        };

        // SAFETY: Area::StartingAt{}.to_rect is guaranteed to be the entire plane
        unsafe { Self::from_slice_unsafe(plane.data(), geometry, rect) }
    }
}

impl<'a, T: Pixel> PlaneRegionMut<'a, T> {
    pub fn new(plane: &'a mut Plane<T>, rect: Rect) -> Self {
        let geometry = plane.geometry();
        Self::from_slice(plane.data_mut(), geometry, rect)
    }

    #[expect(clippy::needless_pass_by_ref_mut)]
    pub fn data_ptr_mut(&mut self) -> *mut T {
        self.data
    }

    #[expect(clippy::needless_pass_by_ref_mut)]
    pub fn rows_iter_mut(&mut self) -> PlaneRegionRowsIterMut<'_, T> {
        PlaneRegionRowsIterMut {
            data: self.data,
            stride: self.plane_cfg.stride,
            width: self.rect.width,
            remaining: self.rect.height.get(),
            phantom: PhantomData,
        }
    }

    pub fn as_const(&self) -> PlaneRegion<'_, T> {
        PlaneRegion {
            data: self.data,
            plane_cfg: self.plane_cfg,
            rect: self.rect,
            phantom: PhantomData,
        }
    }
}

impl<T: Pixel> IndexMut<usize> for PlaneRegionMut<'_, T> {
    fn index_mut(&mut self, index: usize) -> &mut Self::Output {
        assert!(index < self.rect.height.get());
        // SAFETY: The above assert ensures we do not access OOB data.
        unsafe {
            let ptr = self.data.add(index * self.plane_cfg.stride.get());
            slice::from_raw_parts_mut(ptr, self.rect.width.get())
        }
    }
}

/// Iterator over plane region rows
pub struct PlaneRegionRowsIter<'a, T: Pixel> {
    data: *const T,
    stride: NonZeroUsize,
    width: NonZeroUsize,
    remaining: usize,
    phantom: PhantomData<&'a T>,
}

impl<'a, T: Pixel> Iterator for PlaneRegionRowsIter<'a, T> {
    type Item = &'a [T];

    fn next(&mut self) -> Option<Self::Item> {
        (self.remaining > 0).then(|| {
            // SAFETY: struct ensures we do not overflow bounds
            let row = unsafe {
                let ptr = self.data;
                self.data = self.data.add(self.stride.get());
                slice::from_raw_parts(ptr, self.width.get())
            };
            self.remaining -= 1;
            row
        })
    }

    fn size_hint(&self) -> (usize, Option<usize>) {
        (self.remaining, Some(self.remaining))
    }
}

/// Mutable iterator over plane region rows
pub struct PlaneRegionRowsIterMut<'a, T: Pixel> {
    data: *mut T,
    stride: NonZeroUsize,
    width: NonZeroUsize,
    remaining: usize,
    phantom: PhantomData<&'a mut T>,
}

impl<'a, T: Pixel> Iterator for PlaneRegionRowsIterMut<'a, T> {
    type Item = &'a mut [T];

    fn next(&mut self) -> Option<Self::Item> {
        (self.remaining > 0).then(|| {
            // SAFETY: struct ensures we do not overflow bounds
            let row = unsafe {
                let ptr = self.data;
                self.data = self.data.add(self.stride.get());
                slice::from_raw_parts_mut(ptr, self.width.get())
            };
            self.remaining -= 1;
            row
        })
    }

    fn size_hint(&self) -> (usize, Option<usize>) {
        (self.remaining, Some(self.remaining))
    }
}

impl<T: Pixel> ExactSizeIterator for PlaneRegionRowsIter<'_, T> {
}
impl<T: Pixel> FusedIterator for PlaneRegionRowsIter<'_, T> {
}
impl<T: Pixel> ExactSizeIterator for PlaneRegionRowsIterMut<'_, T> {
}
impl<T: Pixel> FusedIterator for PlaneRegionRowsIterMut<'_, T> {
}

pub struct VertWindows<'a, T: Pixel> {
    data: *const T,
    plane_cfg: PlaneGeometry,
    remaining: usize,
    output_rect: Rect,
    phantom: PhantomData<&'a T>,
}

pub struct HorzWindows<'a, T: Pixel> {
    data: *const T,
    plane_cfg: PlaneGeometry,
    remaining: usize,
    output_rect: Rect,
    phantom: PhantomData<&'a T>,
}

impl<'a, T: Pixel> Iterator for VertWindows<'a, T> {
    type Item = PlaneRegion<'a, T>;

    fn next(&mut self) -> Option<Self::Item> {
        self.nth(0)
    }

    fn size_hint(&self) -> (usize, Option<usize>) {
        (self.remaining, Some(self.remaining))
    }

    fn nth(&mut self, n: usize) -> Option<Self::Item> {
        (self.remaining > n).then(|| {
            // SAFETY: struct ensures we do not overflow bounds
            self.data = unsafe { self.data.add(self.plane_cfg.stride.get() * n) };
            self.output_rect.y += n as isize;
            let output = PlaneRegion {
                data: self.data,
                plane_cfg: self.plane_cfg,
                rect: self.output_rect,
                phantom: PhantomData,
            };
            // SAFETY: We verified that we have enough data left to not go OOB.
            self.data = unsafe { self.data.add(self.plane_cfg.stride.get()) };
            self.output_rect.y += 1;
            self.remaining -= n + 1;
            output
        })
    }
}

impl<'a, T: Pixel> Iterator for HorzWindows<'a, T> {
    type Item = PlaneRegion<'a, T>;

    fn next(&mut self) -> Option<Self::Item> {
        self.nth(0)
    }

    fn size_hint(&self) -> (usize, Option<usize>) {
        (self.remaining, Some(self.remaining))
    }

    fn nth(&mut self, n: usize) -> Option<Self::Item> {
        (self.remaining > n).then(|| {
            // SAFETY: struct ensures we do not overflow bounds
            self.data = unsafe { self.data.add(n) };
            self.output_rect.x += n as isize;
            let output = PlaneRegion {
                data: self.data,
                plane_cfg: self.plane_cfg,
                rect: self.output_rect,
                phantom: PhantomData,
            };
            // SAFETY: We verified that we have enough data left to not go OOB.
            self.data = unsafe { self.data.add(1) };
            self.output_rect.x += 1;
            self.remaining -= n + 1;
            output
        })
    }
}

impl<T: Pixel> ExactSizeIterator for VertWindows<'_, T> {
}
impl<T: Pixel> FusedIterator for VertWindows<'_, T> {
}
impl<T: Pixel> ExactSizeIterator for HorzWindows<'_, T> {
}
impl<T: Pixel> FusedIterator for HorzWindows<'_, T> {
}

/// Rectangle of a plane region, in pixels
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Rect {
    // coordinates relative to the plane origin (pad_left, pad_top)
    pub x: isize,
    pub y: isize,
    pub width: NonZeroUsize,
    pub height: NonZeroUsize,
}

// Structure to describe a rectangle area in several ways
//
// To retrieve a subregion from a region, we need to provide the subregion
// bounds, relative to its parent region. The subregion must always be included
// in its parent region.
//
// For that purpose, we could just use a rectangle (x, y, width, height), but
// this would be too cumbersome to use in practice. For example, we often need
// to pass a subregion from an offset, using the same bottom-right corner as
// its parent, or to pass a subregion expressed in block offset instead of
// pixel offset.
//
// Area provides a flexible way to describe a subregion.
#[derive(Debug, Clone, Copy)]
pub enum Area {
    /// A well-defined rectangle
    Rect(Rect),
    /// A rectangle starting at offset (x, y) and ending at the bottom-right
    /// corner of the parent
    StartingAt { x: isize, y: isize },
    /// a rectangle starting at given block offset until the bottom-right corner
    /// of the parent
    BlockStartingAt { bo: BlockOffset },
}

impl Area {
    /// Convert to a rectangle of pixels.
    /// For a `BlockRect` and `BlockStartingAt`, for subsampled chroma planes,
    /// the returned rect will be aligned to a 4x4 chroma block.
    /// This is necessary for `compute_distortion` and `rdo_cfl_alpha` as
    /// the subsampled chroma block covers multiple luma blocks.
    pub const fn to_rect(
        self,
        xdec: usize,
        ydec: usize,
        parent_width: NonZeroUsize,
        parent_height: NonZeroUsize,
    ) -> Rect {
        match self {
            Area::Rect(rect) => rect,
            Area::StartingAt { x, y } => Rect {
                x,
                y,
                width: NonZeroUsize::new((parent_width.get() as isize - x) as usize)
                    .expect("cannot be zero"),
                height: NonZeroUsize::new((parent_height.get() as isize - y) as usize)
                    .expect("cannot be zero"),
            },
            Area::BlockStartingAt { bo } => {
                let x = (bo.x >> xdec << BLOCK_TO_PLANE_SHIFT) as isize;
                let y = (bo.y >> ydec << BLOCK_TO_PLANE_SHIFT) as isize;
                Rect {
                    x,
                    y,
                    width: NonZeroUsize::new((parent_width.get() as isize - x) as usize)
                        .expect("cannot be zero"),
                    height: NonZeroUsize::new((parent_height.get() as isize - y) as usize)
                        .expect("cannot be zero"),
                }
            }
        }
    }
}

pub trait AsRegion<T: Pixel> {
    fn as_region(&self) -> PlaneRegion<'_, T>;
    fn region(&self, area: Area) -> PlaneRegion<'_, T>;
    fn region_mut(&mut self, area: Area) -> PlaneRegionMut<'_, T>;
}

impl<T: Pixel> AsRegion<T> for Plane<T> {
    fn as_region(&self) -> PlaneRegion<'_, T> {
        PlaneRegion::new_from_plane(self)
    }

    fn region(&self, area: Area) -> PlaneRegion<'_, T> {
        let geometry = self.geometry();
        let rect = area.to_rect(
            geometry.subsampling_x.get() as usize >> 1,
            geometry.subsampling_y.get() as usize >> 1,
            NonZeroUsize::new(geometry.stride.get() - geometry.pad_left).expect("cannot be zero"),
            NonZeroUsize::new(geometry.alloc_height().get() - geometry.pad_top)
                .expect("cannot be zero"),
        );
        PlaneRegion::new(self, rect)
    }

    fn region_mut(&mut self, area: Area) -> PlaneRegionMut<'_, T> {
        let geometry = self.geometry();
        let rect = area.to_rect(
            geometry.subsampling_x.get() as usize >> 1,
            geometry.subsampling_y.get() as usize >> 1,
            NonZeroUsize::new(geometry.stride.get() - geometry.pad_left).expect("cannot be zero"),
            NonZeroUsize::new(geometry.alloc_height().get() - geometry.pad_top)
                .expect("cannot be zero"),
        );
        PlaneRegionMut::new(self, rect)
    }
}

/// Absolute offset in blocks inside a plane, where a block is defined
/// to be an `N*N` square where `N == (1 << BLOCK_TO_PLANE_SHIFT)`.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct PlaneBlockOffset(pub BlockOffset);

impl PlaneBlockOffset {
    /// Convert to plane offset without decimation.
    pub const fn to_luma_plane_offset(self) -> PlaneOffset {
        self.0.to_luma_plane_offset()
    }
}

// A Plane, PlaneSlice, or PlaneRegion is assumed to include or be able to
// include padding on the edge of the frame
#[derive(Clone, Copy)]
pub struct PlaneSlice<'a, T: Pixel> {
    pub plane: &'a Plane<T>,
    pub x: isize,
    pub y: isize,
}

impl<'a, T: Pixel> PlaneSlice<'a, T> {
    pub fn as_ptr(&self) -> *const T {
        self[0].as_ptr()
    }

    pub fn clamp(&self) -> PlaneSlice<'a, T> {
        PlaneSlice {
            plane: self.plane,
            x: self.x.clamp(
                -(self.plane.geometry().pad_left as isize),
                self.plane.geometry().width.get() as isize,
            ),
            y: self.y.clamp(
                -(self.plane.geometry().pad_top as isize),
                self.plane.geometry().height.get() as isize,
            ),
        }
    }

    pub fn subslice(&self, xo: usize, yo: usize) -> PlaneSlice<'a, T> {
        PlaneSlice {
            plane: self.plane,
            x: self.x + xo as isize,
            y: self.y + yo as isize,
        }
    }

    /// A slice starting i pixels above the current one.
    #[allow(dead_code)]
    pub fn go_up(&self, i: usize) -> PlaneSlice<'a, T> {
        PlaneSlice {
            plane: self.plane,
            x: self.x,
            y: self.y - i as isize,
        }
    }

    /// A slice starting i pixels to the left of the current one.
    #[allow(dead_code)]
    pub fn go_left(&self, i: usize) -> PlaneSlice<'a, T> {
        PlaneSlice {
            plane: self.plane,
            x: self.x - i as isize,
            y: self.y,
        }
    }

    /// Checks if `add_y` and `add_x` lies in the allocated bounds of the
    /// underlying plane.
    pub fn accessible(&self, add_x: usize, add_y: usize) -> bool {
        let y = (self.y + add_y as isize + self.plane.geometry().pad_top as isize) as usize;
        let x = (self.x + add_x as isize + self.plane.geometry().pad_left as isize) as usize;
        y < self.plane.geometry().alloc_height().get() && x < self.plane.geometry().stride.get()
    }

    /// Checks if -`sub_x` and -`sub_y` lies in the allocated bounds of the
    /// underlying plane.
    pub fn accessible_neg(&self, sub_x: usize, sub_y: usize) -> bool {
        let y = self.y - sub_y as isize + self.plane.geometry().pad_top as isize;
        let x = self.x - sub_x as isize + self.plane.geometry().pad_left as isize;
        y >= 0 && x >= 0
    }
}

impl<T: Pixel> Index<usize> for PlaneSlice<'_, T> {
    type Output = [T];

    fn index(&self, index: usize) -> &Self::Output {
        let range = row_range(self.plane.geometry(), self.x, self.y + index as isize);
        &self.plane.data()[range]
    }
}

pub(crate) fn plane_to_plane_slice<T: Pixel>(
    plane: &Plane<T>,
    po: PlaneOffset,
) -> PlaneSlice<'_, T> {
    PlaneSlice {
        plane,
        x: po.x,
        y: po.y,
    }
}

/// This version of the function includes the padding on the right side of the
/// image
fn row_range(geometry: PlaneGeometry, x: isize, y: isize) -> Range<usize> {
    debug_assert!(geometry.pad_top as isize + y >= 0);
    debug_assert!(geometry.pad_left as isize + x >= 0);

    let base_y = (geometry.pad_top as isize + y) as usize;
    let base_x = (geometry.pad_left as isize + x) as usize;
    let base = base_y * geometry.stride.get() + base_x;
    let width = geometry.stride.get() - base_x;
    base..base + width
}

/// Returns a plane downscaled from the source plane by a factor of `scale` (not
/// padded)
pub(crate) fn downscale<T: Pixel, const SCALE: usize>(
    plane: &Plane<T>,
    bit_depth: NonZeroU8,
) -> Plane<T> {
    let new_frame = FrameBuilder::new(
        NonZeroUsize::new(plane.width().get() / SCALE)
            .expect("cannot downscale a plane with width < SCALE"),
        NonZeroUsize::new(plane.height().get() / SCALE)
            .expect("cannot downscale a plane with height < SCALE"),
        v_frame::chroma::ChromaSubsampling::Monochrome,
        bit_depth,
    )
    .build()
    .expect("should be able to build new frame");
    let mut new_plane = new_frame.y_plane;

    downscale_in_place::<T, SCALE>(plane, &mut new_plane);

    new_plane
}

/// Downscales the source plane by a factor of `scale`, writing the result to
/// `in_plane` (not padded)
///
/// # Panics
///
/// - If the current plane's width and height are not at least `SCALE` times the
///   `in_plane`'s
pub(crate) fn downscale_in_place<T: Pixel, const SCALE: usize>(
    plane: &Plane<T>,
    in_plane: &mut Plane<T>,
) {
    let stride = in_plane.geometry().stride.get();
    let width = in_plane.width().get();
    let height = in_plane.height().get();

    assert!(width * SCALE <= plane.geometry().stride.get() - plane.geometry().pad_left);
    assert!(height * SCALE <= plane.geometry().alloc_height().get() - plane.geometry().pad_top);

    // SAFETY: Bounds checks have been removed for performance reasons
    unsafe {
        let src = plane;
        let box_pixels = SCALE * SCALE;
        let half_box_pixels = box_pixels as u32 / 2; // Used for rounding int division

        let data_origin = &src.data()[src.data_origin()..];
        let plane_data_mut_slice = in_plane.data_mut();

        let src_stride = src.geometry().stride.get();
        // Iter dst rows
        for row_idx in 0..height {
            let dst_row = plane_data_mut_slice.get_unchecked_mut(row_idx * stride..);
            // Iter dst cols
            for (col_idx, dst) in dst_row.get_unchecked_mut(..width).iter_mut().enumerate() {
                macro_rules! generate_inner_loop {
                    ($x:ty) => {
                        let mut sum = half_box_pixels as $x;
                        // Sum box of size scale * scale

                        // Iter src row
                        for y in 0..SCALE {
                            let src_row_idx = row_idx * SCALE + y;
                            let src_row = data_origin.get_unchecked((src_row_idx * src_stride)..);

                            // Iter src col
                            for x in 0..SCALE {
                                let src_col_idx = col_idx * SCALE + x;
                                pastey::paste! {
                                    sum += src_row.get_unchecked(src_col_idx).[<to_ $x>]().expect("value should fit into integer");
                                }
                            }
                        }

                        // Box average
                        let avg = sum as usize / box_pixels;
                        *dst = T::from(avg).expect("value should fit into Pixel");
                    };
                }

                // Use 16 bit precision if overflow would not happen
                if size_of::<T>() == 1
                    && SCALE as u128 * SCALE as u128 * (u8::MAX as u128) + half_box_pixels as u128
                        <= u16::MAX as u128
                {
                    generate_inner_loop!(u16);
                } else {
                    generate_inner_loop!(u32);
                }
            }
        }
    }
}
