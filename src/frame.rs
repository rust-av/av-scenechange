use crate::pred::{FilterMode, PredictionMode};
use crate::refs::*;
use std::fmt::Debug;
use std::marker::PhantomData;
use std::ops::{Index, IndexMut};
use std::sync::Arc;
use std::{fmt, mem, slice};
use v_frame::frame::Frame;
use v_frame::math::Fixed;
use v_frame::pixel::{ChromaSampling, Pixel};
use v_frame::plane::{Plane, PlaneConfig, PlaneOffset};

pub const SB_SIZE_LOG2: usize = 6;
pub const MI_SIZE_LOG2: usize = 2;
pub const MIB_SIZE_LOG2: usize = (SB_SIZE_LOG2 - MI_SIZE_LOG2);
pub const MI_SIZE: usize = (1 << MI_SIZE_LOG2);
pub const MAX_SB_SIZE_LOG2: usize = 7;
pub const MAX_MIB_SIZE_LOG2: usize = (MAX_SB_SIZE_LOG2 - MI_SIZE_LOG2);

pub const BLOCK_TO_PLANE_SHIFT: usize = MI_SIZE_LOG2;
pub const SUPERBLOCK_TO_BLOCK_SHIFT: usize = MIB_SIZE_LOG2;

pub const MAX_TX_SIZE: usize = 64;

pub trait AsRegion<T: Pixel> {
    fn as_region(&self) -> PlaneRegion<'_, T>;
    fn as_region_mut(&mut self) -> PlaneRegionMut<'_, T>;
    fn region_mut(&mut self, area: Area) -> PlaneRegionMut<'_, T>;
    fn region(&self, area: Area) -> PlaneRegion<'_, T>;
}

impl<T: Pixel> AsRegion<T> for Plane<T> {
    #[inline(always)]
    fn region(&self, area: Area) -> PlaneRegion<'_, T> {
        let rect = area.to_rect(
            self.cfg.stride - self.cfg.xorigin as usize,
            self.cfg.alloc_height - self.cfg.yorigin as usize,
        );
        PlaneRegion::new(self, rect)
    }

    #[inline(always)]
    fn region_mut(&mut self, area: Area) -> PlaneRegionMut<'_, T> {
        let rect = area.to_rect(
            self.cfg.stride - self.cfg.xorigin as usize,
            self.cfg.alloc_height - self.cfg.yorigin as usize,
        );
        PlaneRegionMut::new(self, rect)
    }

    #[inline(always)]
    fn as_region(&self) -> PlaneRegion<'_, T> {
        self.region(Area::StartingAt { x: 0, y: 0 })
    }

    #[inline(always)]
    fn as_region_mut(&mut self) -> PlaneRegionMut<'_, T> {
        self.region_mut(Area::StartingAt { x: 0, y: 0 })
    }
}

/// Possible types of a frame.
#[allow(dead_code, non_camel_case_types)]
#[derive(Debug, Eq, PartialEq, Clone, Copy)]
pub enum FrameType {
    /// Key frame.
    KEY,
    /// Inter-frame.
    INTER,
    /// Intra-only frame.
    INTRA_ONLY,
    /// Switching frame.
    SWITCH,
}

impl fmt::Display for FrameType {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        use self::FrameType::*;
        match self {
            KEY => write!(f, "Key frame"),
            INTER => write!(f, "Inter frame"),
            INTRA_ONLY => write!(f, "Intra only frame"),
            SWITCH => write!(f, "Switching frame"),
        }
    }
}

#[derive(Debug, Clone)]
pub struct FrameInvariants<T: Pixel> {
    pub bit_depth: usize,
    pub width: usize,
    pub height: usize,
    pub chroma_sampling: ChromaSampling,
    pub sb_width: usize,
    pub sb_height: usize,
    pub mi_width: usize,
    pub mi_height: usize,
    pub w_in_b: usize,
    pub h_in_b: usize,
    pub input_frameno: u64,
    pub allow_high_precision_mv: bool,
    pub me_lambda: f64,
    pub frame_type: FrameType,
    pub primary_ref_frame: u32,
    pub ref_frames: [u8; INTER_REFS_PER_FRAME],
    pub ref_frame_sign_bias: [bool; INTER_REFS_PER_FRAME],
    pub rec_buffer: ReferenceFramesSet<T>,
    pub me_range_scale: u8,
    pub default_filter: FilterMode,
}

impl<T: Pixel> FrameInvariants<T> {
    fn new(width: usize, height: usize, bit_depth: usize, chroma_sampling: ChromaSampling) -> Self {
        assert!(
            bit_depth <= mem::size_of::<T>() * 8,
            "bit depth cannot fit into u8"
        );

        let w_in_b = 2 * width.align_power_of_two_and_shift(3); // MiCols, ((width+7)/8)<<3 >> MI_SIZE_LOG2
        let h_in_b = 2 * height.align_power_of_two_and_shift(3); // MiRows, ((height+7)/8)<<3 >> MI_SIZE_LOG2

        Self {
            bit_depth,
            width,
            height,
            chroma_sampling,
            sb_width: width.align_power_of_two_and_shift(6),
            sb_height: height.align_power_of_two_and_shift(6),
            w_in_b,
            h_in_b,
            mi_width: width >> MI_SIZE_LOG2,
            mi_height: height >> MI_SIZE_LOG2,
            input_frameno: 0,
            allow_high_precision_mv: false,
            frame_type: FrameType::KEY,
            primary_ref_frame: PRIMARY_REF_NONE,
            ref_frames: [0; INTER_REFS_PER_FRAME],
            ref_frame_sign_bias: [false; INTER_REFS_PER_FRAME],
            rec_buffer: ReferenceFramesSet::new(),
            me_lambda: 0.0,
            me_range_scale: 1,
            default_filter: FilterMode::REGULAR,
        }
    }

    pub fn new_key_frame(
        width: usize,
        height: usize,
        bit_depth: usize,
        chroma_sampling: ChromaSampling,
        input_frameno: u64,
    ) -> Self {
        let mut fi = Self::new(width, height, bit_depth, chroma_sampling);
        fi.input_frameno = input_frameno;
        fi
    }

    /// Returns the created FrameInvariants along with a bool indicating success.
    /// This interface provides simpler usage, because we always need the produced
    /// FrameInvariants regardless of success or failure.
    pub(crate) fn new_inter_frame(previous_fi: &Self) -> Self {
        let mut fi = previous_fi.clone();
        fi.frame_type = FrameType::INTER;

        let ref_in_previous_group = RefType::LAST3_FRAME;
        fi.primary_ref_frame = (ref_in_previous_group.to_index()) as u32;

        fi.input_frameno += 1;
        fi
    }

    #[inline(always)]
    pub fn sb_size_log2(&self) -> usize {
        6
    }
}

#[derive(Debug, Clone)]
pub struct FrameState<'a, T: Pixel> {
    pub sb_size_log2: usize,
    pub input: &'a Frame<T>,
    pub input_hres: Plane<T>, // half-resolution version of input luma
    pub input_qres: Plane<T>, // quarter-resolution version of input luma
    pub frame_mvs: Arc<Vec<FrameMotionVectors>>,
}

impl<'a, T: Pixel> FrameState<'a, T> {
    pub fn new_with_frame(fi: &FrameInvariants<T>, frame: &'a Frame<T>) -> Self {
        let luma_width = frame.planes[0].cfg.width;
        let luma_height = frame.planes[0].cfg.height;

        let hres = frame.planes[0].downsampled(luma_width / 2, luma_height / 2);
        let qres = hres.downsampled(luma_width / 4, luma_height / 4);

        Self {
            sb_size_log2: fi.sb_size_log2(),
            input: frame,
            input_hres: hres,
            input_qres: qres,
            frame_mvs: {
                let mut vec = Vec::with_capacity(REF_FRAMES);
                for _ in 0..REF_FRAMES {
                    vec.push(FrameMotionVectors::new(fi.w_in_b, fi.h_in_b));
                }
                Arc::new(vec)
            },
        }
    }
}

/// Bounded region of a plane
///
/// This allows to give access to a rectangular area of a plane without
/// giving access to the whole plane.
#[derive(Debug)]
pub struct PlaneRegion<'a, T: Pixel> {
    data: *const T, // points to (plane_cfg.x, plane_cfg.y)
    pub plane_cfg: &'a PlaneConfig,
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
    pub plane_cfg: &'a PlaneConfig,
    rect: Rect,
    phantom: PhantomData<&'a mut T>,
}

// common impl for PlaneRegion and PlaneRegionMut
macro_rules! plane_region_common {
  // $name: PlaneRegion or PlaneRegionMut
  // $as_ptr: as_ptr or as_mut_ptr
  // $opt_mut: nothing or mut
  ($name:ident, $as_ptr:ident $(,$opt_mut:tt)?) => {
    #[allow(unused)]
    impl<'a, T: Pixel> $name<'a, T> {
      #[inline(always)]
      pub fn from_slice(data: &'a $($opt_mut)? [T], cfg: &'a PlaneConfig, rect:
        Rect) -> Self {
        assert!(rect.x >= -(cfg.xorigin as isize));
        assert!(rect.y >= -(cfg.yorigin as isize));
        assert!(cfg.xorigin as isize + rect.x + rect.width as isize <= cfg.stride as isize);
        assert!(cfg.yorigin as isize + rect.y + rect.height as isize <= cfg.alloc_height as isize);
        let origin = (cfg.yorigin as isize + rect.y) * cfg.stride as isize
                    + cfg.xorigin as isize + rect.x;
        Self {
          data: unsafe { data.$as_ptr().offset(origin) },
          plane_cfg: cfg,
          rect,
          phantom: PhantomData,
        }
      }
      #[inline(always)]
      pub fn new(plane: &'a $($opt_mut)? Plane<T>, rect: Rect) -> Self {
        Self::from_slice(& $($opt_mut)? plane.data, &plane.cfg, rect)
      }

      #[inline(always)]
      pub fn data_ptr(&self) -> *const T {
        self.data
      }

      #[inline(always)]
      pub fn rect(&self) -> &Rect {
        &self.rect
      }

      #[inline(always)]
      pub fn rows_iter(&self) -> RegionRowsIter<'_, T> {
        RegionRowsIter {
          data: self.data,
          stride: self.plane_cfg.stride,
          width: self.rect.width,
          remaining: self.rect.height,
          phantom: PhantomData,
        }
      }

      pub fn vert_windows(&self, h: usize) -> VertWindows<'_, T> {
        VertWindows {
          data: self.data,
          plane_cfg: self.plane_cfg,
          remaining: (self.rect.height as isize - h as isize + 1).max(0) as usize,
          output_rect: Rect {
            x: self.rect.x,
            y: self.rect.y,
            width: self.rect.width,
            height: h
          }
        }
      }

      pub fn horz_windows(&self, w: usize) -> HorzWindows<'_, T> {
        HorzWindows {
          data: self.data,
          plane_cfg: self.plane_cfg,
          remaining: (self.rect.width as isize - w as isize + 1).max(0) as usize,
          output_rect: Rect {
            x: self.rect.x,
            y: self.rect.y,
            width: w,
            height: self.rect.height
          }
        }
      }

      // Return a view to a subregion of the plane
      //
      // The subregion must be included in (i.e. must not exceed) this region.
      //
      // It is described by an `Area`, relative to this region.
      //
      // # Example
      //
      // ``` ignore
      // # use rav1e::tiling::*;
      // # fn f(region: &PlaneRegion<'_, u16>) {
      // // a subregion from (10, 8) to the end of the region
      // let subregion = region.subregion(Area::StartingAt { x: 10, y: 8 });
      // # }
      // ```
      //
      // ``` ignore
      // # use rav1e::context::*;
      // # use rav1e::tiling::*;
      // # fn f(region: &PlaneRegion<'_, u16>) {
      // // a subregion from the top-left of block (2, 3) having size (64, 64)
      // let bo = BlockOffset { x: 2, y: 3 };
      // let subregion = region.subregion(Area::BlockRect { bo, width: 64, height: 64 });
      // # }
      // ```
      #[inline(always)]
      pub fn subregion(&self, area: Area) -> PlaneRegion<'_, T> {
        let rect = area.to_rect(
          self.rect.width,
          self.rect.height,
        );
        assert!(rect.x >= 0 && rect.x as usize <= self.rect.width);
        assert!(rect.y >= 0 && rect.y as usize <= self.rect.height);
        let data = unsafe {
          self.data.add(rect.y as usize * self.plane_cfg.stride + rect.x as usize)
        };
        let absolute_rect = Rect {
          x: self.rect.x + rect.x,
          y: self.rect.y + rect.y,
          width: rect.width,
          height: rect.height,
        };
        PlaneRegion {
          data,
          plane_cfg: &self.plane_cfg,
          rect: absolute_rect,
          phantom: PhantomData,
        }
      }

      #[inline(always)]
      pub fn to_frame_plane_offset(&self, tile_po: PlaneOffset) -> PlaneOffset {
        PlaneOffset {
          x: self.rect.x + tile_po.x,
          y: self.rect.y + tile_po.y,
        }
      }
    }

    unsafe impl<T: Pixel> Send for $name<'_, T> {}
    unsafe impl<T: Pixel> Sync for $name<'_, T> {}

    impl<T: Pixel> Index<usize> for $name<'_, T> {
      type Output = [T];

      #[inline(always)]
      fn index(&self, index: usize) -> &Self::Output {
        assert!(index < self.rect.height);
        unsafe {
          let ptr = self.data.add(index * self.plane_cfg.stride);
          slice::from_raw_parts(ptr, self.rect.width)
        }
      }
    }
  }
}

plane_region_common!(PlaneRegion, as_ptr);
plane_region_common!(PlaneRegionMut, as_mut_ptr, mut);

impl<'a, T: Pixel> PlaneRegionMut<'a, T> {
    #[inline(always)]
    pub fn data_ptr_mut(&mut self) -> *mut T {
        self.data
    }

    #[inline(always)]
    pub fn rows_iter_mut(&mut self) -> RegionRowsIterMut<'_, T> {
        RegionRowsIterMut {
            data: self.data,
            stride: self.plane_cfg.stride,
            width: self.rect.width,
            remaining: self.rect.height,
            phantom: PhantomData,
        }
    }

    #[inline(always)]
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
    #[inline(always)]
    fn index_mut(&mut self, index: usize) -> &mut Self::Output {
        assert!(index < self.rect.height);
        unsafe {
            let ptr = self.data.add(index * self.plane_cfg.stride);
            slice::from_raw_parts_mut(ptr, self.rect.width)
        }
    }
}

/// Iterator over plane region rows
pub struct RegionRowsIter<'a, T: Pixel> {
    data: *const T,
    stride: usize,
    width: usize,
    remaining: usize,
    phantom: PhantomData<&'a T>,
}

impl<'a, T: Pixel> Iterator for RegionRowsIter<'a, T> {
    type Item = &'a [T];

    #[inline(always)]
    fn next(&mut self) -> Option<Self::Item> {
        if self.remaining > 0 {
            let row = unsafe {
                let ptr = self.data;
                self.data = self.data.add(self.stride);
                slice::from_raw_parts(ptr, self.width)
            };
            self.remaining -= 1;
            Some(row)
        } else {
            None
        }
    }

    #[inline(always)]
    fn size_hint(&self) -> (usize, Option<usize>) {
        (self.remaining, Some(self.remaining))
    }
}

impl<T: Pixel> ExactSizeIterator for RegionRowsIter<'_, T> {}

/// Mutable iterator over plane region rows
pub struct RegionRowsIterMut<'a, T: Pixel> {
    data: *mut T,
    stride: usize,
    width: usize,
    remaining: usize,
    phantom: PhantomData<&'a mut T>,
}

impl<'a, T: Pixel> Iterator for RegionRowsIterMut<'a, T> {
    type Item = &'a mut [T];

    #[inline(always)]
    fn next(&mut self) -> Option<Self::Item> {
        if self.remaining > 0 {
            let row = unsafe {
                let ptr = self.data;
                self.data = self.data.add(self.stride);
                slice::from_raw_parts_mut(ptr, self.width)
            };
            self.remaining -= 1;
            Some(row)
        } else {
            None
        }
    }

    #[inline(always)]
    fn size_hint(&self) -> (usize, Option<usize>) {
        (self.remaining, Some(self.remaining))
    }
}

impl<T: Pixel> ExactSizeIterator for RegionRowsIterMut<'_, T> {}

/// Rectangle of a plane region, in pixels
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Rect {
    // coordinates relative to the plane origin (xorigin, yorigin)
    pub x: isize,
    pub y: isize,
    pub width: usize,
    pub height: usize,
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
    Rect {
        x: isize,
        y: isize,
        width: usize,
        height: usize,
    },
    /// A rectangle starting at offset (x, y) and ending at the bottom-right
    /// corner of the parent
    StartingAt { x: isize, y: isize },
}

impl Area {
    #[inline(always)]
    /// Convert to a rectangle of pixels.
    /// For a BlockRect and BlockStartingAt, for subsampled chroma planes,
    /// the returned rect will be aligned to a 4x4 chroma block.
    /// This is necessary for compute_distortion and rdo_cfl_alpha as
    /// the subsampled chroma block covers multiple luma blocks.
    pub fn to_rect(&self, parent_width: usize, parent_height: usize) -> Rect {
        match *self {
            Area::Rect {
                x,
                y,
                width,
                height,
            } => Rect {
                x,
                y,
                width,
                height,
            },
            Area::StartingAt { x, y } => Rect {
                x,
                y,
                width: (parent_width as isize - x) as usize,
                height: (parent_height as isize - y) as usize,
            },
        }
    }
}

/// Absolute offset in blocks, where a block is defined
/// to be an N*N square where N = (1 << BLOCK_TO_PLANE_SHIFT).
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct BlockOffset {
    pub x: usize,
    pub y: usize,
}

/// Absolute offset in blocks inside a plane, where a block is defined
/// to be an N*N square where N = (1 << BLOCK_TO_PLANE_SHIFT).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct PlaneBlockOffset(pub BlockOffset);

/// Absolute offset in blocks inside a tile, where a block is defined
/// to be an N*N square where N = (1 << BLOCK_TO_PLANE_SHIFT).
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct TileBlockOffset(pub BlockOffset);

impl BlockOffset {
    /// Convert to plane offset without decimation.
    #[inline(always)]
    const fn to_luma_plane_offset(self) -> PlaneOffset {
        PlaneOffset {
            x: (self.x as isize) << BLOCK_TO_PLANE_SHIFT,
            y: (self.y as isize) << BLOCK_TO_PLANE_SHIFT,
        }
    }
}

impl PlaneBlockOffset {
    /// Convert to plane offset without decimation.
    #[inline(always)]
    pub const fn to_luma_plane_offset(self) -> PlaneOffset {
        self.0.to_luma_plane_offset()
    }
}

/// Absolute offset in superblocks, where a superblock is defined
/// to be an N*N square where N = (1 << SUPERBLOCK_TO_PLANE_SHIFT).
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct SuperBlockOffset {
    pub x: usize,
    pub y: usize,
}

/// Absolute offset in superblocks inside a plane, where a superblock is defined
/// to be an N*N square where N = (1 << SUPERBLOCK_TO_PLANE_SHIFT).
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct PlaneSuperBlockOffset(pub SuperBlockOffset);

/// Absolute offset in superblocks inside a tile, where a superblock is defined
/// to be an N*N square where N = (1 << SUPERBLOCK_TO_PLANE_SHIFT).
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct TileSuperBlockOffset(pub SuperBlockOffset);

impl SuperBlockOffset {
    /// Offset of a block inside the current superblock.
    const fn block_offset(self, block_x: usize, block_y: usize) -> BlockOffset {
        BlockOffset {
            x: (self.x << SUPERBLOCK_TO_BLOCK_SHIFT) + block_x,
            y: (self.y << SUPERBLOCK_TO_BLOCK_SHIFT) + block_y,
        }
    }
}

impl PlaneSuperBlockOffset {
    /// Offset of a block inside the current superblock.
    #[inline(always)]
    pub const fn block_offset(self, block_x: usize, block_y: usize) -> PlaneBlockOffset {
        PlaneBlockOffset(self.0.block_offset(block_x, block_y))
    }
}

pub struct VertWindows<'a, T: Pixel> {
    data: *const T,
    plane_cfg: &'a PlaneConfig,
    remaining: usize,
    output_rect: Rect,
}

pub struct HorzWindows<'a, T: Pixel> {
    data: *const T,
    plane_cfg: &'a PlaneConfig,
    remaining: usize,
    output_rect: Rect,
}

impl<'a, T: Pixel> Iterator for VertWindows<'a, T> {
    type Item = PlaneRegion<'a, T>;

    #[inline(always)]
    #[allow(clippy::iter_nth_zero)]
    fn next(&mut self) -> Option<Self::Item> {
        self.nth(0)
    }

    #[inline(always)]
    fn size_hint(&self) -> (usize, Option<usize>) {
        (self.remaining, Some(self.remaining))
    }

    #[inline(always)]
    fn nth(&mut self, n: usize) -> Option<Self::Item> {
        if self.remaining > n {
            self.data = unsafe { self.data.add(self.plane_cfg.stride * n) };
            self.output_rect.y += n as isize;
            let output = PlaneRegion {
                data: self.data,
                plane_cfg: &self.plane_cfg,
                rect: self.output_rect,
                phantom: PhantomData,
            };
            self.data = unsafe { self.data.add(self.plane_cfg.stride) };
            self.output_rect.y += 1;
            self.remaining -= n + 1;
            Some(output)
        } else {
            None
        }
    }
}

impl<'a, T: Pixel> Iterator for HorzWindows<'a, T> {
    type Item = PlaneRegion<'a, T>;

    #[inline(always)]
    #[allow(clippy::iter_nth_zero)]
    fn next(&mut self) -> Option<Self::Item> {
        self.nth(0)
    }

    #[inline(always)]
    fn size_hint(&self) -> (usize, Option<usize>) {
        (self.remaining, Some(self.remaining))
    }

    #[inline(always)]
    fn nth(&mut self, n: usize) -> Option<Self::Item> {
        if self.remaining > n {
            self.data = unsafe { self.data.add(n) };
            self.output_rect.x += n as isize;
            let output = PlaneRegion {
                data: self.data,
                plane_cfg: &self.plane_cfg,
                rect: self.output_rect,
                phantom: PhantomData,
            };
            self.data = unsafe { self.data.add(1) };
            self.output_rect.x += 1;
            self.remaining -= n + 1;
            Some(output)
        } else {
            None
        }
    }
}

impl<T: Pixel> ExactSizeIterator for VertWindows<'_, T> {}
impl<T: Pixel> ExactSizeIterator for HorzWindows<'_, T> {}

#[allow(non_camel_case_types)]
#[derive(Debug, Copy, Clone, PartialEq, PartialOrd, Ord, Eq)]
pub enum BlockSize {
    BLOCK_4X4,
    BLOCK_4X8,
    BLOCK_8X4,
    BLOCK_8X8,
    BLOCK_8X16,
    BLOCK_16X8,
    BLOCK_16X16,
    BLOCK_16X32,
    BLOCK_32X16,
    BLOCK_32X32,
    BLOCK_32X64,
    BLOCK_64X32,
    BLOCK_64X64,
    BLOCK_64X128,
    BLOCK_128X64,
    BLOCK_128X128,
    BLOCK_4X16,
    BLOCK_16X4,
    BLOCK_8X32,
    BLOCK_32X8,
    BLOCK_16X64,
    BLOCK_64X16,
}

impl BlockSize {
    pub fn from_width_and_height(w: usize, h: usize) -> BlockSize {
        use BlockSize::*;
        match (w, h) {
            (4, 4) => BLOCK_4X4,
            (4, 8) => BLOCK_4X8,
            (8, 4) => BLOCK_8X4,
            (8, 8) => BLOCK_8X8,
            (8, 16) => BLOCK_8X16,
            (16, 8) => BLOCK_16X8,
            (16, 16) => BLOCK_16X16,
            (16, 32) => BLOCK_16X32,
            (32, 16) => BLOCK_32X16,
            (32, 32) => BLOCK_32X32,
            (32, 64) => BLOCK_32X64,
            (64, 32) => BLOCK_64X32,
            (64, 64) => BLOCK_64X64,
            (64, 128) => BLOCK_64X128,
            (128, 64) => BLOCK_128X64,
            (128, 128) => BLOCK_128X128,
            (4, 16) => BLOCK_4X16,
            (16, 4) => BLOCK_16X4,
            (8, 32) => BLOCK_8X32,
            (32, 8) => BLOCK_32X8,
            (16, 64) => BLOCK_16X64,
            (64, 16) => BLOCK_64X16,
            _ => unreachable!(),
        }
    }

    pub fn width(self) -> usize {
        1 << self.width_log2()
    }

    pub fn width_log2(self) -> usize {
        use BlockSize::*;
        match self {
            BLOCK_4X4 | BLOCK_4X8 | BLOCK_4X16 => 2,
            BLOCK_8X4 | BLOCK_8X8 | BLOCK_8X16 | BLOCK_8X32 => 3,
            BLOCK_16X4 | BLOCK_16X8 | BLOCK_16X16 | BLOCK_16X32 | BLOCK_16X64 => 4,
            BLOCK_32X8 | BLOCK_32X16 | BLOCK_32X32 | BLOCK_32X64 => 5,
            BLOCK_64X16 | BLOCK_64X32 | BLOCK_64X64 | BLOCK_64X128 => 6,
            BLOCK_128X64 | BLOCK_128X128 => 7,
        }
    }

    pub fn width_mi(self) -> usize {
        self.width() >> MI_SIZE_LOG2
    }

    pub fn height(self) -> usize {
        1 << self.height_log2()
    }

    pub fn height_log2(self) -> usize {
        use BlockSize::*;
        match self {
            BLOCK_4X4 | BLOCK_8X4 | BLOCK_16X4 => 2,
            BLOCK_4X8 | BLOCK_8X8 | BLOCK_16X8 | BLOCK_32X8 => 3,
            BLOCK_4X16 | BLOCK_8X16 | BLOCK_16X16 | BLOCK_32X16 | BLOCK_64X16 => 4,
            BLOCK_8X32 | BLOCK_16X32 | BLOCK_32X32 | BLOCK_64X32 => 5,
            BLOCK_16X64 | BLOCK_32X64 | BLOCK_64X64 | BLOCK_128X64 => 6,
            BLOCK_64X128 | BLOCK_128X128 => 7,
        }
    }

    pub fn height_mi(self) -> usize {
        self.height() >> MI_SIZE_LOG2
    }

    pub fn tx_size(self) -> TxSize {
        use BlockSize::*;
        use TxSize::*;
        match self {
            BLOCK_4X4 => TX_4X4,
            BLOCK_4X8 => TX_4X8,
            BLOCK_8X4 => TX_8X4,
            BLOCK_8X8 => TX_8X8,
            BLOCK_8X16 => TX_8X16,
            BLOCK_16X8 => TX_16X8,
            BLOCK_16X16 => TX_16X16,
            BLOCK_16X32 => TX_16X32,
            BLOCK_32X16 => TX_32X16,
            BLOCK_32X32 => TX_32X32,
            BLOCK_32X64 => TX_32X64,
            BLOCK_64X32 => TX_64X32,
            BLOCK_4X16 => TX_4X16,
            BLOCK_16X4 => TX_16X4,
            BLOCK_8X32 => TX_8X32,
            BLOCK_32X8 => TX_32X8,
            BLOCK_16X64 => TX_16X64,
            BLOCK_64X16 => TX_64X16,
            _ => TX_64X64,
        }
    }
}

/// Transform Size
#[allow(non_camel_case_types)]
#[derive(Copy, Clone, Debug, PartialEq, PartialOrd, Eq, Ord)]
pub enum TxSize {
    TX_4X4,
    TX_8X8,
    TX_16X16,
    TX_32X32,
    TX_64X64,

    TX_4X8,
    TX_8X4,
    TX_8X16,
    TX_16X8,
    TX_16X32,
    TX_32X16,
    TX_32X64,
    TX_64X32,

    TX_4X16,
    TX_16X4,
    TX_8X32,
    TX_32X8,
    TX_16X64,
    TX_64X16,
}

impl TxSize {
    pub fn width(self) -> usize {
        1 << self.width_log2()
    }

    pub fn width_log2(self) -> usize {
        use TxSize::*;
        match self {
            TX_4X4 | TX_4X8 | TX_4X16 => 2,
            TX_8X8 | TX_8X4 | TX_8X16 | TX_8X32 => 3,
            TX_16X16 | TX_16X8 | TX_16X32 | TX_16X4 | TX_16X64 => 4,
            TX_32X32 | TX_32X16 | TX_32X64 | TX_32X8 => 5,
            TX_64X64 | TX_64X32 | TX_64X16 => 6,
        }
    }

    pub fn width_mi(self) -> usize {
        self.width() >> MI_SIZE_LOG2
    }

    pub fn height(self) -> usize {
        1 << self.height_log2()
    }

    pub fn height_log2(self) -> usize {
        use TxSize::*;
        match self {
            TX_4X4 | TX_8X4 | TX_16X4 => 2,
            TX_8X8 | TX_4X8 | TX_16X8 | TX_32X8 => 3,
            TX_16X16 | TX_8X16 | TX_32X16 | TX_4X16 | TX_64X16 => 4,
            TX_32X32 | TX_16X32 | TX_64X32 | TX_8X32 => 5,
            TX_64X64 | TX_32X64 | TX_16X64 => 6,
        }
    }

    pub fn height_mi(self) -> usize {
        self.height() >> MI_SIZE_LOG2
    }
}

pub trait Dim {
    const W: usize;
    const H: usize;
}

macro_rules! blocks_dimension {
    ($(($W:expr, $H:expr)),+) => {
        paste::item! {
            $(
                pub struct [<Block $W x $H>];

                impl Dim for [<Block $W x $H>] {
                    const W: usize = $W;
                    const H: usize = $H;
                }
            )*
        }
    };
}

blocks_dimension! {
    (4, 4), (8, 8), (16, 16), (32, 32), (64, 64),
    (4, 8), (8, 16), (16, 32), (32, 64),
    (8, 4), (16, 8), (32, 16), (64, 32),
    (4, 16), (8, 32), (16, 64),
    (16, 4), (32, 8), (64, 16)
}

#[derive(Clone)]
pub struct Block {
    pub mode: PredictionMode,
    pub partition: PartitionType,
    pub skip: bool,
    pub ref_frames: [RefType; 2],
    pub mv: [MotionVector; 2],
    // note: indexes are reflist index, NOT the same as libaom
    pub neighbors_ref_counts: [usize; INTER_REFS_PER_FRAME],
    pub cdef_index: u8,
    pub bsize: BlockSize,
    pub n4_w: usize, /* block width in the unit of mode_info */
    pub n4_h: usize, /* block height in the unit of mode_info */
    pub txsize: TxSize,
}

impl Default for Block {
    fn default() -> Block {
        Block {
            mode: PredictionMode::DC_PRED,
            partition: PartitionType::PARTITION_NONE,
            skip: false,
            ref_frames: [RefType::INTRA_FRAME; 2],
            mv: [MotionVector::default(); 2],
            neighbors_ref_counts: [0; INTER_REFS_PER_FRAME],
            cdef_index: 0,
            bsize: BlockSize::BLOCK_64X64,
            n4_w: BlockSize::BLOCK_64X64.width_mi(),
            n4_h: BlockSize::BLOCK_64X64.height_mi(),
            txsize: TxSize::TX_64X64,
        }
    }
}

#[allow(non_camel_case_types)]
#[derive(Copy, Clone, PartialEq, PartialOrd, Debug)]
pub enum PartitionType {
    PARTITION_NONE,
}
