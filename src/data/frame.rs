use std::{num::NonZeroUsize, sync::Arc};

use v_frame::{frame::Frame, pixel::Pixel, plane::Plane};

use crate::{
    data::motion::{RefMEStats, ReferenceFramesSet},
    math::Fixed,
};

pub const ALLOWED_REF_FRAMES: &[RefType] = &[RefType::LAST_FRAME];
pub const INTER_REFS_PER_FRAME: usize = 7;

#[derive(Clone)]
pub struct FrameState<T: Pixel> {
    pub input: Arc<Frame<T>>,
    pub input_hres: Option<Arc<Plane<T>>>, // half-resolution version of input luma
    pub input_qres: Option<Arc<Plane<T>>>, // quarter-resolution version of input luma
    pub frame_me_stats: RefMEStats,
}

impl<T: Pixel> FrameState<T> {
    /// Similar to [`FrameState::new_with_frame`], but takes an `me_stats`
    /// and `rec` to enable reusing the same underlying allocations to create
    /// a `FrameState`
    ///
    /// This function primarily exists for [`estimate_inter_costs`], and so
    /// it does not create hres or qres versions of `frame` as downscaling is
    /// somewhat expensive and are not needed for [`estimate_inter_costs`].
    pub fn new_with_frame_and_me_stats_and_rec(frame: Arc<Frame<T>>, me_stats: RefMEStats) -> Self {
        Self {
            input: frame,
            input_hres: None,
            input_qres: None,
            frame_me_stats: me_stats,
        }
    }
}

// LAST_FRAME through ALTREF_FRAME correspond to slots 0-6.
#[allow(non_camel_case_types)]
#[allow(dead_code)]
#[derive(PartialEq, Eq, PartialOrd, Copy, Clone, Debug)]
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
    /// convert to a ref list index, 0-6 (`INTER_REFS_PER_FRAME`)
    ///
    /// # Panics
    ///
    /// - If the ref type is a None or Intra frame
    pub fn to_index(self) -> usize {
        use self::RefType::{INTRA_FRAME, NONE_FRAME};

        match self {
            NONE_FRAME => {
                panic!("Tried to get slot of NONE_FRAME");
            }
            INTRA_FRAME => {
                panic!("Tried to get slot of INTRA_FRAME");
            }
            _ => (self as usize) - 1,
        }
    }
}

// Frame Invariants are invariant inside a frame
#[derive(Clone)]
pub struct FrameInvariants<T: Pixel> {
    pub w_in_b: NonZeroUsize,
    pub h_in_b: NonZeroUsize,
    pub ref_frames: [u8; INTER_REFS_PER_FRAME],
    pub rec_buffer: ReferenceFramesSet<T>,
}

impl<T: Pixel> FrameInvariants<T> {
    pub fn new(width: NonZeroUsize, height: NonZeroUsize) -> Self {
        // MiCols, ((width+7)/8)<<3 >> MI_SIZE_LOG2
        let w_in_b = NonZeroUsize::new(2 * width.get().align_power_of_two_and_shift(3))
            .expect("cannot be zero");
        // MiRows, ((height+7)/8)<<3 >> MI_SIZE_LOG2
        let h_in_b = NonZeroUsize::new(2 * height.get().align_power_of_two_and_shift(3))
            .expect("cannot be zero");

        Self {
            w_in_b,
            h_in_b,
            ref_frames: [0; INTER_REFS_PER_FRAME],
            rec_buffer: ReferenceFramesSet::new(),
        }
    }

    pub fn new_key_frame(width: NonZeroUsize, height: NonZeroUsize) -> Self {
        Self::new(width, height)
    }

    /// Returns the created `FrameInvariants`, or `None` if this should be
    /// a placeholder frame.
    pub(crate) fn new_inter_frame(
        previous_coded_fi: &Self,
        output_frameno_in_gop: u64,
    ) -> Option<Self> {
        let mut fi = previous_coded_fi.clone();
        let idx_in_group_output = get_idx_in_group_output(output_frameno_in_gop);
        let order_hint = get_order_hint(output_frameno_in_gop, idx_in_group_output);

        // this is the slot that the current frame is going to be saved into
        let slot_idx = get_slot_idx(0, order_hint);

        // level 0 has no forward references
        // default to last P frame
        //
        // calculations done relative to the slot_idx for this frame.
        // the last four frames can be found by subtracting from the current
        //
        // add 4 to prevent underflow
        // TODO: maybe use order_hint here like in get_slot_idx?
        // this is the previous P frame
        fi.ref_frames = [(slot_idx + 4 - 1) as u8 % 4; INTER_REFS_PER_FRAME];

        Some(fi)
    }
}

// All the stuff below is ripped from InterCfg but assumes
// reordering and multiref are off, so pyramid depth is always 0
const fn get_slot_idx(level: u64, order_hint: u32) -> u32 {
    // Frames with level == 0 are stored in slots 0..4, and frames with higher
    //  values of level in slots 4..8
    if level == 0 {
        order_hint & 3
    } else {
        // This only works with pyramid_depth <= 4.
        3 + level as u32
    }
}

/// Get the index of an output frame in its re-ordering group given the output
///  frame number of the frame in the current keyframe gop.
/// When re-ordering is disabled, this always returns 0.
fn get_idx_in_group_output(output_frameno_in_gop: u64) -> u64 {
    // The first frame in the GOP should be a keyframe and is not re-ordered,
    // so we should not be calling this function on it.
    debug_assert!(output_frameno_in_gop > 0);

    output_frameno_in_gop - 1
}

/// Get the order-hint of an output frame given the output frame number of the
///  frame in the current keyframe gop and the index of that output frame
///  in its re-ordering gorup.
fn get_order_hint(output_frameno_in_gop: u64, idx_in_group_output: u64) -> u32 {
    // The first frame in the GOP should be a keyframe, but currently this
    //  function only handles inter frames.
    // We could return 0 for keyframes if keyframe support is needed.
    debug_assert!(output_frameno_in_gop > 0);

    // Which P-frame group in the current gop is this output frame in?
    // Subtract 1 because the first frame in the gop is always a keyframe.
    let group_idx = output_frameno_in_gop - 1;
    // Get the offset to the corresponding input frame.
    let offset = idx_in_group_output + 1;
    // Construct the final order hint relative to the start of the group.
    (group_idx + offset) as u32
}
