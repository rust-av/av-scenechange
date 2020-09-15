use std::ops;
use std::ops::{Index, IndexMut};
use std::sync::Arc;
use v_frame::frame::Frame;
use v_frame::pixel::Pixel;
use v_frame::plane::Plane;

pub const PRIMARY_REF_NONE: u32 = 7;
pub const INTER_REFS_PER_FRAME: usize = 7;
pub const REF_FRAMES_LOG2: usize = 3;
pub const REF_FRAMES: usize = 1 << REF_FRAMES_LOG2;

pub const ALL_INTER_REFS: [RefType; 7] = [
    RefType::LAST_FRAME,
    RefType::LAST2_FRAME,
    RefType::LAST3_FRAME,
    RefType::GOLDEN_FRAME,
    RefType::BWDREF_FRAME,
    RefType::ALTREF2_FRAME,
    RefType::ALTREF_FRAME,
];

// LAST_FRAME through ALTREF_FRAME correspond to slots 0-6.
#[allow(non_camel_case_types)]
#[derive(PartialEq, Eq, PartialOrd, Copy, Clone)]
pub enum RefType {
    INTRA_FRAME = 0,
    LAST_FRAME = 1,
    LAST2_FRAME = 2,
    LAST3_FRAME = 3,
    GOLDEN_FRAME = 4,
    BWDREF_FRAME = 5,
    ALTREF2_FRAME = 6,
    ALTREF_FRAME = 7,
    NONE_FRAME = 8,
}

impl RefType {
    // convert to a ref list index, 0-6 (INTER_REFS_PER_FRAME)
    pub fn to_index(self) -> usize {
        match self {
            RefType::NONE_FRAME => {
                panic!("Tried to get slot of NONE_FRAME");
            }
            RefType::INTRA_FRAME => {
                panic!("Tried to get slot of INTRA_FRAME");
            }
            _ => (self as usize) - 1,
        }
    }
}

#[derive(Debug, Clone)]
pub struct ReferenceFrame<T: Pixel> {
    pub order_hint: u32,
    pub frame: Arc<Frame<T>>,
    pub input_hres: Arc<Plane<T>>,
    pub input_qres: Arc<Plane<T>>,
    pub frame_mvs: Arc<Vec<FrameMotionVectors>>,
    pub output_frameno: u64,
}

#[derive(Debug, Clone, Default)]
pub struct ReferenceFramesSet<T: Pixel> {
    pub frames: [Option<Arc<ReferenceFrame<T>>>; REF_FRAMES as usize],
}

impl<T: Pixel> ReferenceFramesSet<T> {
    pub fn new() -> Self {
        Self {
            frames: Default::default(),
        }
    }
}

#[derive(Debug, Clone)]
pub struct FrameMotionVectors {
    mvs: Box<[MotionVector]>,
    pub cols: usize,
    pub rows: usize,
}

impl FrameMotionVectors {
    pub fn new(cols: usize, rows: usize) -> Self {
        Self {
            // dynamic allocation: once per frame
            mvs: vec![MotionVector::default(); cols * rows].into_boxed_slice(),
            cols,
            rows,
        }
    }
}

impl Index<usize> for FrameMotionVectors {
    type Output = [MotionVector];
    #[inline]
    fn index(&self, index: usize) -> &Self::Output {
        &self.mvs[index * self.cols..(index + 1) * self.cols]
    }
}

impl IndexMut<usize> for FrameMotionVectors {
    #[inline]
    fn index_mut(&mut self, index: usize) -> &mut Self::Output {
        &mut self.mvs[index * self.cols..(index + 1) * self.cols]
    }
}

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct MotionVector {
    pub row: i16,
    pub col: i16,
}

impl ops::Add<MotionVector> for MotionVector {
    type Output = MotionVector;

    fn add(self, _rhs: MotionVector) -> MotionVector {
        MotionVector {
            row: self.row + _rhs.row,
            col: self.col + _rhs.col,
        }
    }
}

impl ops::Div<i16> for MotionVector {
    type Output = MotionVector;

    fn div(self, _rhs: i16) -> MotionVector {
        MotionVector {
            row: self.row / _rhs,
            col: self.col / _rhs,
        }
    }
}

impl MotionVector {
    pub const fn quantize_to_fullpel(self) -> Self {
        Self {
            row: (self.row / 8) * 8,
            col: (self.col / 8) * 8,
        }
    }

    pub fn is_zero(self) -> bool {
        self.row == 0 && self.col == 0
    }
}
