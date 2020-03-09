#![allow(non_upper_case_globals)]

#[cfg(not(target_arch = "x86_64"))]
pub use native::*;
#[cfg(target_arch = "x86_64")]
pub use x86::*;

use crate::frame::BlockSize::*;
use crate::frame::*;
use crate::mc::*;
use crate::refs::{MotionVector, RefType};
use crate::util::AlignedArray;
use std::mem::size_of;
use v_frame::pixel::Pixel;
use v_frame::plane::{Plane, PlaneOffset, PlaneSlice};

#[allow(non_camel_case_types)]
#[derive(Copy, Clone, Debug, PartialEq, PartialOrd, Eq, Ord)]
pub enum PredictionMode {
    DC_PRED,
    V_PRED,    // Vertical
    H_PRED,    // Horizontal
    D45_PRED,  // Directional 45  deg = round(arctan(1/1) * 180/pi)
    D135_PRED, // Directional 135 deg = 180 - 45
    D117_PRED, // Directional 117 deg = 180 - 63
    D153_PRED, // Directional 153 deg = 180 - 27
    D207_PRED, // Directional 207 deg = 180 + 27
    D63_PRED,  // Directional 63  deg = round(arctan(2/1) * 180/pi)
    PAETH_PRED,
    UV_CFL_PRED,
    NEWMV,
}

impl PredictionMode {
    pub fn predict_intra<T: Pixel>(
        self,
        rect: Rect,
        dst: &mut PlaneRegionMut<'_, T>,
        tx_size: TxSize,
        bit_depth: usize,
        edge_buf: &AlignedArray<[T; 4 * MAX_TX_SIZE + 1]>,
    ) {
        assert!(self == PredictionMode::DC_PRED);
        let &Rect {
            x: frame_x,
            y: frame_y,
            ..
        } = dst.rect();
        debug_assert!(frame_x >= 0 && frame_y >= 0);
        // x and y are expressed relative to the tile
        let x = frame_x - rect.x;
        let y = frame_y - rect.y;

        let variant = PredictionVariant::new(x as usize, y as usize);

        dispatch_predict_intra::<T>(self, variant, dst, tx_size, bit_depth, edge_buf);
    }

    pub fn predict_inter<T: Pixel>(
        self,
        fi: &FrameInvariants<T>,
        p: usize,
        frame_po: PlaneOffset,
        dst: &mut PlaneRegionMut<'_, T>,
        width: usize,
        height: usize,
        ref_frames: [RefType; 2],
        mvs: [MotionVector; 2],
    ) {
        assert!(self == PredictionMode::NEWMV);
        let mode = fi.default_filter;
        let is_compound =
            ref_frames[1] != RefType::INTRA_FRAME && ref_frames[1] != RefType::NONE_FRAME;

        fn get_params<'a, T: Pixel>(
            rec_plane: &'a Plane<T>,
            po: PlaneOffset,
            mv: MotionVector,
        ) -> (i32, i32, PlaneSlice<'a, T>) {
            let rec_cfg = &rec_plane.cfg;
            let shift_row = 3 + rec_cfg.ydec;
            let shift_col = 3 + rec_cfg.xdec;
            let row_offset = mv.row as i32 >> shift_row;
            let col_offset = mv.col as i32 >> shift_col;
            let row_frac = (mv.row as i32 - (row_offset << shift_row)) << (4 - shift_row);
            let col_frac = (mv.col as i32 - (col_offset << shift_col)) << (4 - shift_col);
            let qo = PlaneOffset {
                x: po.x + col_offset as isize - 3,
                y: po.y + row_offset as isize - 3,
            };
            (
                row_frac,
                col_frac,
                rec_plane.slice(qo).clamp().subslice(3, 3),
            )
        };

        if !is_compound {
            if let Some(ref rec) =
                fi.rec_buffer.frames[fi.ref_frames[ref_frames[0].to_index()] as usize]
            {
                let (row_frac, col_frac, src) = get_params(&rec.frame.planes[p], frame_po, mvs[0]);
                put_8tap(
                    dst,
                    src,
                    width,
                    height,
                    col_frac,
                    row_frac,
                    mode,
                    mode,
                    fi.bit_depth,
                );
            }
        } else {
            let mut tmp: [AlignedArray<[i16; 128 * 128]>; 2] =
                [AlignedArray::uninitialized(), AlignedArray::uninitialized()];
            for i in 0..2 {
                if let Some(ref rec) =
                    fi.rec_buffer.frames[fi.ref_frames[ref_frames[i].to_index()] as usize]
                {
                    let (row_frac, col_frac, src) =
                        get_params(&rec.frame.planes[p], frame_po, mvs[i]);
                    prep_8tap(
                        &mut tmp[i].array,
                        src,
                        width,
                        height,
                        col_frac,
                        row_frac,
                        mode,
                        mode,
                        fi.bit_depth,
                    );
                }
            }
            mc_avg(
                dst,
                &tmp[0].array,
                &tmp[1].array,
                width,
                height,
                fi.bit_depth,
            );
        }
    }
}

#[derive(Copy, Clone, Debug)]
pub enum PredictionVariant {
    None,
    Left,
    Top,
    Both,
}

impl PredictionVariant {
    fn new(x: usize, y: usize) -> Self {
        match (x, y) {
            (0, 0) => PredictionVariant::None,
            (_, 0) => PredictionVariant::Left,
            (0, _) => PredictionVariant::Top,
            _ => PredictionVariant::Both,
        }
    }
}

#[derive(Copy, Clone, Debug, PartialEq, PartialOrd)]
pub enum FilterMode {
    REGULAR = 0,
    SMOOTH = 1,
    SHARP = 2,
    BILINEAR = 3,
}

#[allow(clippy::cognitive_complexity)]
pub fn get_intra_edges<T: Pixel>(
    dst: &PlaneRegion<'_, T>,
    partition_bo: PlaneBlockOffset,
    bx: usize,
    by: usize,
    partition_size: BlockSize,
    po: PlaneOffset,
    tx_size: TxSize,
    bit_depth: usize,
    opt_mode: Option<PredictionMode>,
) -> AlignedArray<[T; 4 * MAX_TX_SIZE + 1]> {
    let plane_cfg = &dst.plane_cfg;

    let mut edge_buf: AlignedArray<[T; 4 * MAX_TX_SIZE + 1]> = AlignedArray::uninitialized();
    let base = 128u16 << (bit_depth - 8);
    {
        // left pixels are order from bottom to top and right-aligned
        let (left, not_left) = edge_buf.array.split_at_mut(2 * MAX_TX_SIZE);
        let (top_left, above) = not_left.split_at_mut(1);

        let x = po.x as usize;
        let y = po.y as usize;

        let mut needs_left = true;
        let mut needs_topleft = true;
        let mut needs_top = true;
        let mut needs_topright = true;
        let mut needs_bottomleft = true;

        if let Some(mut mode) = opt_mode {
            mode = match mode {
                PredictionMode::PAETH_PRED => match (x, y) {
                    (0, 0) => PredictionMode::DC_PRED,
                    (_, 0) => PredictionMode::H_PRED,
                    (0, _) => PredictionMode::V_PRED,
                    _ => PredictionMode::PAETH_PRED,
                },
                _ => mode,
            };

            let dc_or_cfl = mode == PredictionMode::DC_PRED || mode == PredictionMode::UV_CFL_PRED;

            needs_left = mode != PredictionMode::V_PRED
                && (!dc_or_cfl || x != 0)
                && !(mode == PredictionMode::D45_PRED || mode == PredictionMode::D63_PRED);
            needs_topleft = mode == PredictionMode::PAETH_PRED
                || mode == PredictionMode::D117_PRED
                || mode == PredictionMode::D135_PRED
                || mode == PredictionMode::D153_PRED;
            needs_top = mode != PredictionMode::H_PRED && (!dc_or_cfl || y != 0);
            needs_topright = mode == PredictionMode::D45_PRED || mode == PredictionMode::D63_PRED;
            needs_bottomleft = mode == PredictionMode::D207_PRED;
        }

        // Needs left
        if needs_left {
            if x != 0 {
                for i in 0..tx_size.height() {
                    left[2 * MAX_TX_SIZE - tx_size.height() + i] =
                        dst[y + tx_size.height() - 1 - i][x - 1];
                }
            } else {
                let val = if y != 0 {
                    dst[y - 1][0]
                } else {
                    T::cast_from(base + 1)
                };
                for v in left[2 * MAX_TX_SIZE - tx_size.height()..].iter_mut() {
                    *v = val
                }
            }
        }

        // Needs top-left
        if needs_topleft {
            top_left[0] = match (x, y) {
                (0, 0) => T::cast_from(base),
                (_, 0) => dst[0][x - 1],
                (0, _) => dst[y - 1][0],
                _ => dst[y - 1][x - 1],
            };
        }

        // Needs top
        if needs_top {
            if y != 0 {
                above[..tx_size.width()].copy_from_slice(&dst[y - 1][x..x + tx_size.width()]);
            } else {
                let val = if x != 0 {
                    dst[0][x - 1]
                } else {
                    T::cast_from(base - 1)
                };
                for v in above[..tx_size.width()].iter_mut() {
                    *v = val;
                }
            }
        }

        let bx4 = bx * (tx_size.width() >> MI_SIZE_LOG2); // bx,by are in tx block indices
        let by4 = by * (tx_size.height() >> MI_SIZE_LOG2);

        let have_top = by4 != 0
            || if plane_cfg.ydec != 0 {
                partition_bo.0.y > 1
            } else {
                partition_bo.0.y > 0
            };
        let have_left = bx4 != 0
            || if plane_cfg.xdec != 0 {
                partition_bo.0.x > 1
            } else {
                partition_bo.0.x > 0
            };

        let right_available = x + tx_size.width() < dst.rect().width;
        let bottom_available = y + tx_size.height() < dst.rect().height;

        let scaled_partition_size =
            supersample_chroma_bsize(partition_size, plane_cfg.xdec, plane_cfg.ydec);

        // Needs top right
        if needs_topright {
            debug_assert!(plane_cfg.xdec <= 1 && plane_cfg.ydec <= 1);

            let num_avail = if y != 0
                && has_top_right(
                    scaled_partition_size,
                    partition_bo,
                    have_top,
                    right_available,
                    tx_size,
                    by4,
                    bx4,
                    plane_cfg.xdec,
                    plane_cfg.ydec,
                ) {
                tx_size.width().min(dst.rect().width - x - tx_size.width())
            } else {
                0
            };
            if num_avail > 0 {
                above[tx_size.width()..tx_size.width() + num_avail].copy_from_slice(
                    &dst[y - 1][x + tx_size.width()..x + tx_size.width() + num_avail],
                );
            }
            if num_avail < tx_size.height() {
                let val = above[tx_size.width() + num_avail - 1];
                for v in above[tx_size.width() + num_avail..tx_size.width() + tx_size.height()]
                    .iter_mut()
                {
                    *v = val;
                }
            }
        }

        // Needs bottom left
        if needs_bottomleft {
            debug_assert!(plane_cfg.xdec <= 1 && plane_cfg.ydec <= 1);

            let num_avail = if x != 0
                && has_bottom_left(
                    scaled_partition_size,
                    partition_bo,
                    bottom_available,
                    have_left,
                    tx_size,
                    by4,
                    bx4,
                    plane_cfg.xdec,
                    plane_cfg.ydec,
                ) {
                tx_size
                    .height()
                    .min(dst.rect().height - y - tx_size.height())
            } else {
                0
            };
            if num_avail > 0 {
                for i in 0..num_avail {
                    left[2 * MAX_TX_SIZE - tx_size.height() - 1 - i] =
                        dst[y + tx_size.height() + i][x - 1];
                }
            }
            if num_avail < tx_size.width() {
                let val = left[2 * MAX_TX_SIZE - tx_size.height() - num_avail];
                for v in left[(2 * MAX_TX_SIZE - tx_size.height() - tx_size.width())
                    ..(2 * MAX_TX_SIZE - tx_size.height() - num_avail)]
                    .iter_mut()
                {
                    *v = val;
                }
            }
        }
    }
    edge_buf
}

fn supersample_chroma_bsize(bsize: BlockSize, ss_x: usize, ss_y: usize) -> BlockSize {
    debug_assert!(ss_x < 2);
    debug_assert!(ss_y < 2);
    match bsize {
        BLOCK_4X4 => match (ss_x, ss_y) {
            (1, 1) => BLOCK_8X8,
            (1, 0) => BLOCK_8X4,
            (0, 1) => BLOCK_4X8,
            _ => bsize,
        },
        BLOCK_4X8 => match (ss_x, ss_y) {
            (1, 1) => BLOCK_8X8,
            (1, 0) => BLOCK_8X8,
            (0, 1) => BLOCK_4X8,
            _ => bsize,
        },
        BLOCK_8X4 => match (ss_x, ss_y) {
            (1, 1) => BLOCK_8X8,
            (1, 0) => BLOCK_8X4,
            (0, 1) => BLOCK_8X8,
            _ => bsize,
        },
        BLOCK_4X16 => match (ss_x, ss_y) {
            (1, 1) => BLOCK_8X16,
            (1, 0) => BLOCK_8X16,
            (0, 1) => BLOCK_4X16,
            _ => bsize,
        },
        BLOCK_16X4 => match (ss_x, ss_y) {
            (1, 1) => BLOCK_16X8,
            (1, 0) => BLOCK_16X4,
            (0, 1) => BLOCK_16X8,
            _ => bsize,
        },
        _ => bsize,
    }
}

pub fn has_top_right(
    /*const AV1_COMMON *cm,*/
    bsize: BlockSize,
    partition_bo: PlaneBlockOffset,
    top_available: bool,
    right_available: bool,
    tx_size: TxSize,
    row_off: usize,
    col_off: usize,
    ss_x: usize,
    _ss_y: usize,
) -> bool {
    if !top_available || !right_available {
        return false;
    };
    let bw_unit = bsize.width_mi();
    let plane_bw_unit = (bw_unit >> ss_x).max(1);
    let top_right_count_unit = tx_size.width_mi();
    let mi_col = partition_bo.0.x;
    let mi_row = partition_bo.0.y;
    if row_off > 0 {
        // Just need to check if enough pixels on the right.
        // 128x128 SB is not supported yet by rav1e
        if bsize.width() > BLOCK_64X64.width() {
            // Special case: For 128x128 blocks, the transform unit whose
            // top-right corner is at the center of the block does in fact have
            // pixels available at its top-right corner.
            if row_off == BLOCK_64X64.height_mi() >> _ss_y
                && col_off + top_right_count_unit == BLOCK_64X64.width_mi() >> ss_x
            {
                return false;
            }
            let plane_bw_unit_64 = BLOCK_64X64.width_mi() >> ss_x;
            let col_off_64 = col_off % plane_bw_unit_64;
            return col_off_64 + top_right_count_unit < plane_bw_unit_64;
        }
        col_off + top_right_count_unit < plane_bw_unit
    } else {
        // All top-right pixels are in the block above, which is already available.
        if col_off + top_right_count_unit < plane_bw_unit {
            return true;
        };

        let bw_in_mi_log2 = bsize.width_log2() - MI_SIZE_LOG2;
        let bh_in_mi_log2 = bsize.height_log2() - MI_SIZE_LOG2;
        //const int sb_mi_size = mi_size_high[cm->seq_params.sb_size];
        let sb_mi_size: usize = 16; // 64x64, fi.sequence.use_128x128_superblock
        let blk_row_in_sb = (mi_row & (sb_mi_size - 1)) >> bh_in_mi_log2;
        let blk_col_in_sb = (mi_col & (sb_mi_size - 1)) >> bw_in_mi_log2;

        // Top row of superblock: so top-right pixels are in the top and/or
        // top-right superblocks, both of which are already available.
        if blk_row_in_sb == 0 {
            return true;
        };

        // Rightmost column of superblock (and not the top row): so top-right pixels
        // fall in the right superblock, which is not available yet.
        if ((blk_col_in_sb + 1) << bw_in_mi_log2) >= sb_mi_size {
            return false;
        };

        // General case (neither top row nor rightmost column): check if the
        // top-right block is coded before the current block.
        let this_blk_index = (blk_row_in_sb << (MAX_MIB_SIZE_LOG2 - bw_in_mi_log2)) + blk_col_in_sb;
        let idx1 = this_blk_index / 8;
        let idx2 = this_blk_index % 8;
        let has_tr_table: &[u8] = get_has_tr_table(bsize);

        ((has_tr_table[idx1] >> idx2) & 1) != 0
    }
}

fn get_has_tr_table(bsize: BlockSize) -> &'static [u8] {
    has_tr_tables[bsize as usize]
}

static has_tr_4x4: &[u8] = &[
    255, 255, 255, 255, 85, 85, 85, 85, 119, 119, 119, 119, 85, 85, 85, 85, 127, 127, 127, 127, 85,
    85, 85, 85, 119, 119, 119, 119, 85, 85, 85, 85, 255, 127, 255, 127, 85, 85, 85, 85, 119, 119,
    119, 119, 85, 85, 85, 85, 127, 127, 127, 127, 85, 85, 85, 85, 119, 119, 119, 119, 85, 85, 85,
    85, 255, 255, 255, 127, 85, 85, 85, 85, 119, 119, 119, 119, 85, 85, 85, 85, 127, 127, 127, 127,
    85, 85, 85, 85, 119, 119, 119, 119, 85, 85, 85, 85, 255, 127, 255, 127, 85, 85, 85, 85, 119,
    119, 119, 119, 85, 85, 85, 85, 127, 127, 127, 127, 85, 85, 85, 85, 119, 119, 119, 119, 85, 85,
    85, 85,
];

static has_tr_4x8: &[u8] = &[
    255, 255, 255, 255, 119, 119, 119, 119, 127, 127, 127, 127, 119, 119, 119, 119, 255, 127, 255,
    127, 119, 119, 119, 119, 127, 127, 127, 127, 119, 119, 119, 119, 255, 255, 255, 127, 119, 119,
    119, 119, 127, 127, 127, 127, 119, 119, 119, 119, 255, 127, 255, 127, 119, 119, 119, 119, 127,
    127, 127, 127, 119, 119, 119, 119,
];

static has_tr_8x4: &[u8] = &[
    255, 255, 0, 0, 85, 85, 0, 0, 119, 119, 0, 0, 85, 85, 0, 0, 127, 127, 0, 0, 85, 85, 0, 0, 119,
    119, 0, 0, 85, 85, 0, 0, 255, 127, 0, 0, 85, 85, 0, 0, 119, 119, 0, 0, 85, 85, 0, 0, 127, 127,
    0, 0, 85, 85, 0, 0, 119, 119, 0, 0, 85, 85, 0, 0,
];

static has_tr_8x8: &[u8] = &[
    255, 255, 85, 85, 119, 119, 85, 85, 127, 127, 85, 85, 119, 119, 85, 85, 255, 127, 85, 85, 119,
    119, 85, 85, 127, 127, 85, 85, 119, 119, 85, 85,
];
static has_tr_8x16: &[u8] = &[
    255, 255, 119, 119, 127, 127, 119, 119, 255, 127, 119, 119, 127, 127, 119, 119,
];
static has_tr_16x8: &[u8] = &[255, 0, 85, 0, 119, 0, 85, 0, 127, 0, 85, 0, 119, 0, 85, 0];
static has_tr_16x16: &[u8] = &[255, 85, 119, 85, 127, 85, 119, 85];
static has_tr_16x32: &[u8] = &[255, 119, 127, 119];
static has_tr_32x16: &[u8] = &[15, 5, 7, 5];

//pub static has_tr_32x32: [u8; 2] = [ 95, 87 ];
static has_tr_32x32: &[u8] = &[95, 87];

static has_tr_32x64: &[u8] = &[127];
static has_tr_64x32: &[u8] = &[19];
static has_tr_64x64: &[u8] = &[7];
static has_tr_64x128: &[u8] = &[3];
static has_tr_128x64: &[u8] = &[1];
static has_tr_128x128: &[u8] = &[1];
static has_tr_4x16: &[u8] = &[
    255, 255, 255, 255, 127, 127, 127, 127, 255, 127, 255, 127, 127, 127, 127, 127, 255, 255, 255,
    127, 127, 127, 127, 127, 255, 127, 255, 127, 127, 127, 127, 127,
];
static has_tr_16x4: &[u8] = &[
    255, 0, 0, 0, 85, 0, 0, 0, 119, 0, 0, 0, 85, 0, 0, 0, 127, 0, 0, 0, 85, 0, 0, 0, 119, 0, 0, 0,
    85, 0, 0, 0,
];
static has_tr_8x32: &[u8] = &[255, 255, 127, 127, 255, 127, 127, 127];
static has_tr_32x8: &[u8] = &[15, 0, 5, 0, 7, 0, 5, 0];
static has_tr_16x64: &[u8] = &[255, 127];
static has_tr_64x16: &[u8] = &[3, 1];

static has_tr_tables: &[&[u8]] = &[
    has_tr_4x4,     // 4x4
    has_tr_4x8,     // 4x8
    has_tr_8x4,     // 8x4
    has_tr_8x8,     // 8x8
    has_tr_8x16,    // 8x16
    has_tr_16x8,    // 16x8
    has_tr_16x16,   // 16x16
    has_tr_16x32,   // 16x32
    has_tr_32x16,   // 32x16
    has_tr_32x32,   // 32x32
    has_tr_32x64,   // 32x64
    has_tr_64x32,   // 64x32
    has_tr_64x64,   // 64x64
    has_tr_64x128,  // 64x128
    has_tr_128x64,  // 128x64
    has_tr_128x128, // 128x128
    has_tr_4x16,    // 4x16
    has_tr_16x4,    // 16x4
    has_tr_8x32,    // 8x32
    has_tr_32x8,    // 32x8
    has_tr_16x64,   // 16x64
    has_tr_64x16,   // 64x16
];

pub fn has_bottom_left(
    /*const AV1_COMMON *cm,*/
    bsize: BlockSize,
    partition_bo: PlaneBlockOffset,
    bottom_available: bool,
    left_available: bool,
    tx_size: TxSize,
    row_off: usize,
    col_off: usize,
    _ss_x: usize,
    ss_y: usize,
) -> bool {
    /*static int has_bottom_left(const AV1_COMMON *cm, BLOCK_SIZE bsize, int mi_row,
    int mi_col, int bottom_available, int left_available,
    PARTITION_TYPE partition, TX_SIZE txsz, int row_off,
    int col_off, int ss_x, int ss_y) {*/
    if !bottom_available || !left_available {
        return false;
    };
    // Special case for 128x* blocks, when col_off is half the block width.
    // This is needed because 128x* superblocks are divided into 64x* blocks in
    // raster order
    // 128x128 SB is not supported yet by rav1e
    if bsize.width() > BLOCK_64X64.width() && col_off > 0 {
        let plane_bw_unit_64 = BLOCK_64X64.width_mi() >> _ss_x;
        let col_off_64 = col_off % plane_bw_unit_64;
        if col_off_64 == 0 {
            // We are at the left edge of top-right or bottom-right 64x* block.
            let plane_bh_unit_64 = BLOCK_64X64.height_mi() >> ss_y;
            let row_off_64 = row_off % plane_bh_unit_64;
            let plane_bh_unit = (bsize.height_mi() >> ss_y).min(plane_bh_unit_64);
            // Check if all bottom-left pixels are in the left 64x* block (which is
            // already coded).
            return row_off_64 + tx_size.height_mi() < plane_bh_unit;
        }
    }
    if col_off > 0 {
        // Bottom-left pixels are in the bottom-left block, which is not available.
        false
    } else {
        let bh_unit = bsize.height_mi();
        let plane_bh_unit = (bh_unit >> ss_y).max(1);
        let bottom_left_count_unit = tx_size.height_mi();

        let mi_col = partition_bo.0.x;
        let mi_row = partition_bo.0.y;

        // All bottom-left pixels are in the left block, which is already available.
        if row_off + bottom_left_count_unit < plane_bh_unit {
            return true;
        };

        let bw_in_mi_log2 = bsize.width_log2() - MI_SIZE_LOG2;
        let bh_in_mi_log2 = bsize.height_log2() - MI_SIZE_LOG2;
        //const int sb_mi_size = mi_size_high[cm->seq_params.sb_size];
        let sb_mi_size: usize = 16; // 64x64, fi.sequence.use_128x128_superblock
        let blk_row_in_sb = (mi_row & (sb_mi_size - 1)) >> bh_in_mi_log2;
        let blk_col_in_sb = (mi_col & (sb_mi_size - 1)) >> bw_in_mi_log2;

        // Leftmost column of superblock: so bottom-left pixels maybe in the left
        // and/or bottom-left superblocks. But only the left superblock is
        // available, so check if all required pixels fall in that superblock.
        if blk_col_in_sb == 0 {
            let blk_start_row_off = blk_row_in_sb << bh_in_mi_log2 >> ss_y;
            let row_off_in_sb = blk_start_row_off + row_off;
            let sb_height_unit = sb_mi_size >> ss_y;
            return row_off_in_sb + bottom_left_count_unit < sb_height_unit;
            //return row_off_in_sb + (bottom_left_count_unit << 1) < sb_height_unit;  // Don't it need tx height? again?
        }

        // Bottom row of superblock (and not the leftmost column): so bottom-left
        // pixels fall in the bottom superblock, which is not available yet.
        if ((blk_row_in_sb + 1) << bh_in_mi_log2) >= sb_mi_size {
            return false;
        };

        // General case (neither leftmost column nor bottom row): check if the
        // bottom-left block is coded before the current block.
        let this_blk_index = (blk_row_in_sb << (MAX_MIB_SIZE_LOG2 - bw_in_mi_log2)) + blk_col_in_sb;
        let idx1 = this_blk_index / 8;
        let idx2 = this_blk_index % 8;
        let has_bl_table: &[u8] = get_has_bl_table(bsize);

        ((has_bl_table[idx1] >> idx2) & 1) != 0
    }
}

static has_bl_4x4: &[u8] = &[
    84, 85, 85, 85, 16, 17, 17, 17, 84, 85, 85, 85, 0, 1, 1, 1, 84, 85, 85, 85, 16, 17, 17, 17, 84,
    85, 85, 85, 0, 0, 1, 0, 84, 85, 85, 85, 16, 17, 17, 17, 84, 85, 85, 85, 0, 1, 1, 1, 84, 85, 85,
    85, 16, 17, 17, 17, 84, 85, 85, 85, 0, 0, 0, 0, 84, 85, 85, 85, 16, 17, 17, 17, 84, 85, 85, 85,
    0, 1, 1, 1, 84, 85, 85, 85, 16, 17, 17, 17, 84, 85, 85, 85, 0, 0, 1, 0, 84, 85, 85, 85, 16, 17,
    17, 17, 84, 85, 85, 85, 0, 1, 1, 1, 84, 85, 85, 85, 16, 17, 17, 17, 84, 85, 85, 85, 0, 0, 0, 0,
];
static has_bl_4x8: &[u8] = &[
    16, 17, 17, 17, 0, 1, 1, 1, 16, 17, 17, 17, 0, 0, 1, 0, 16, 17, 17, 17, 0, 1, 1, 1, 16, 17, 17,
    17, 0, 0, 0, 0, 16, 17, 17, 17, 0, 1, 1, 1, 16, 17, 17, 17, 0, 0, 1, 0, 16, 17, 17, 17, 0, 1,
    1, 1, 16, 17, 17, 17, 0, 0, 0, 0,
];
static has_bl_8x4: &[u8] = &[
    254, 255, 84, 85, 254, 255, 16, 17, 254, 255, 84, 85, 254, 255, 0, 1, 254, 255, 84, 85, 254,
    255, 16, 17, 254, 255, 84, 85, 254, 255, 0, 0, 254, 255, 84, 85, 254, 255, 16, 17, 254, 255,
    84, 85, 254, 255, 0, 1, 254, 255, 84, 85, 254, 255, 16, 17, 254, 255, 84, 85, 254, 255, 0, 0,
];
static has_bl_8x8: &[u8] = &[
    84, 85, 16, 17, 84, 85, 0, 1, 84, 85, 16, 17, 84, 85, 0, 0, 84, 85, 16, 17, 84, 85, 0, 1, 84,
    85, 16, 17, 84, 85, 0, 0,
];
static has_bl_8x16: &[u8] = &[16, 17, 0, 1, 16, 17, 0, 0, 16, 17, 0, 1, 16, 17, 0, 0];
static has_bl_16x8: &[u8] = &[
    254, 84, 254, 16, 254, 84, 254, 0, 254, 84, 254, 16, 254, 84, 254, 0,
];
static has_bl_16x16: &[u8] = &[84, 16, 84, 0, 84, 16, 84, 0];
static has_bl_16x32: &[u8] = &[16, 0, 16, 0];
static has_bl_32x16: &[u8] = &[78, 14, 78, 14];
static has_bl_32x32: &[u8] = &[4, 4];
static has_bl_32x64: &[u8] = &[0];
static has_bl_64x32: &[u8] = &[34];
static has_bl_64x64: &[u8] = &[0];
static has_bl_64x128: &[u8] = &[0];
static has_bl_128x64: &[u8] = &[0];
static has_bl_128x128: &[u8] = &[0];
static has_bl_4x16: &[u8] = &[
    0, 1, 1, 1, 0, 0, 1, 0, 0, 1, 1, 1, 0, 0, 0, 0, 0, 1, 1, 1, 0, 0, 1, 0, 0, 1, 1, 1, 0, 0, 0, 0,
];
static has_bl_16x4: &[u8] = &[
    254, 254, 254, 84, 254, 254, 254, 16, 254, 254, 254, 84, 254, 254, 254, 0, 254, 254, 254, 84,
    254, 254, 254, 16, 254, 254, 254, 84, 254, 254, 254, 0,
];
static has_bl_8x32: &[u8] = &[0, 1, 0, 0, 0, 1, 0, 0];
static has_bl_32x8: &[u8] = &[238, 78, 238, 14, 238, 78, 238, 14];
static has_bl_16x64: &[u8] = &[0, 0];
static has_bl_64x16: &[u8] = &[42, 42];

static has_bl_tables: &[&[u8]] = &[
    has_bl_4x4,     // 4x4
    has_bl_4x8,     // 4x8
    has_bl_8x4,     // 8x4
    has_bl_8x8,     // 8x8
    has_bl_8x16,    // 8x16
    has_bl_16x8,    // 16x8
    has_bl_16x16,   // 16x16
    has_bl_16x32,   // 16x32
    has_bl_32x16,   // 32x16
    has_bl_32x32,   // 32x32
    has_bl_32x64,   // 32x64
    has_bl_64x32,   // 64x32
    has_bl_64x64,   // 64x64
    has_bl_64x128,  // 64x128
    has_bl_128x64,  // 128x64
    has_bl_128x128, // 128x128
    has_bl_4x16,    // 4x16
    has_bl_16x4,    // 16x4
    has_bl_8x32,    // 8x32
    has_bl_32x8,    // 32x8
    has_bl_16x64,   // 16x64
    has_bl_64x16,   // 64x16
];

fn get_has_bl_table(bsize: BlockSize) -> &'static [u8] {
    has_bl_tables[bsize as usize]
}

#[cfg(target_arch = "x86_64")]
mod x86 {
    use super::*;

    macro_rules! decl_angular_ipred_fn {
        ($($f:ident),+) => {
            extern {
                $(
                    fn $f(
                        dst: *mut u8, stride: libc::ptrdiff_t, topleft: *const u8,
                        width: libc::c_int, height: libc::c_int,
                    );
                )*
            }
        };
    }

    decl_angular_ipred_fn! {
        scenechangeasm_ipred_dc_avx2,
        scenechangeasm_ipred_dc_ssse3,
        scenechangeasm_ipred_dc_128_avx2,
        scenechangeasm_ipred_dc_128_ssse3,
        scenechangeasm_ipred_dc_left_avx2,
        scenechangeasm_ipred_dc_left_ssse3,
        scenechangeasm_ipred_dc_top_avx2,
        scenechangeasm_ipred_dc_top_ssse3
    }

    #[inline(always)]
    pub fn dispatch_predict_intra<T: Pixel>(
        mode: PredictionMode,
        variant: PredictionVariant,
        dst: &mut PlaneRegionMut<'_, T>,
        tx_size: TxSize,
        bit_depth: usize,
        edge_buf: &AlignedArray<[T; 4 * MAX_TX_SIZE + 1]>,
    ) {
        let call_native = |dst: &mut PlaneRegionMut<'_, T>| {
            native::dispatch_predict_intra(mode, variant, dst, tx_size, bit_depth, edge_buf);
        };

        if size_of::<T>() != 1 {
            return call_native(dst);
        }

        unsafe {
            let dst_ptr = dst.data_ptr_mut() as *mut _;
            let stride = dst.plane_cfg.stride as libc::ptrdiff_t;
            let edge_ptr = edge_buf.array.as_ptr().offset(2 * MAX_TX_SIZE as isize) as *const _;
            let w = tx_size.width() as libc::c_int;
            let h = tx_size.height() as libc::c_int;

            if is_x86_feature_detected!("avx2") {
                match mode {
                    PredictionMode::DC_PRED => {
                        (match variant {
                            PredictionVariant::None => scenechangeasm_ipred_dc_128_avx2,
                            PredictionVariant::Left => scenechangeasm_ipred_dc_left_avx2,
                            PredictionVariant::Top => scenechangeasm_ipred_dc_top_avx2,
                            PredictionVariant::Both => scenechangeasm_ipred_dc_avx2,
                        })(dst_ptr, stride, edge_ptr, w, h);
                    }
                    _ => call_native(dst),
                }
            } else if is_x86_feature_detected!("ssse3") {
                match mode {
                    PredictionMode::DC_PRED => {
                        (match variant {
                            PredictionVariant::None => scenechangeasm_ipred_dc_128_ssse3,
                            PredictionVariant::Left => scenechangeasm_ipred_dc_left_ssse3,
                            PredictionVariant::Top => scenechangeasm_ipred_dc_top_ssse3,
                            PredictionVariant::Both => scenechangeasm_ipred_dc_ssse3,
                        })(dst_ptr, stride, edge_ptr, w, h);
                    }
                    _ => call_native(dst),
                }
            } else {
                call_native(dst);
            }
        }
    }
}

mod native {
    use super::*;
    use crate::frame::Dim;

    macro_rules! impl_intra {
        ($(($W:expr, $H:expr)),+) => {
            paste::item! {
                $(
                    impl<T: Pixel> Intra<T> for crate::frame::[<Block $W x $H>] {}
                )*
                #[inline(always)]
                pub fn dispatch_predict_intra<T: Pixel>(
                    mode: PredictionMode, variant: PredictionVariant,
                    dst: &mut PlaneRegionMut<'_, T>, tx_size: TxSize, bit_depth: usize,
                    edge_buf: &AlignedArray<[T; 4 * MAX_TX_SIZE + 1]>,
                ) {
                    (match tx_size {
                        $(
                            TxSize::[<TX_$W X $H>] => {
                                predict_intra_inner::<crate::frame::[<Block $W x $H>], _>
                            }
                        )*
                    })(mode, variant, dst, bit_depth, edge_buf);
                }
            }
        };
    }

    impl_intra! {
        (4, 4), (8, 8), (16, 16), (32, 32), (64, 64),
        (4, 8), (8, 16), (16, 32), (32, 64),
        (8, 4), (16, 8), (32, 16), (64, 32),
        (4, 16), (8, 32), (16, 64),
        (16, 4), (32, 8), (64, 16)
    }

    #[inline(always)]
    pub fn predict_intra_inner<B: Intra<T>, T: Pixel>(
        mode: PredictionMode,
        variant: PredictionVariant,
        dst: &mut PlaneRegionMut<'_, T>,
        bit_depth: usize,
        edge_buf: &AlignedArray<[T; 4 * MAX_TX_SIZE + 1]>,
    ) {
        // left pixels are order from bottom to top and right-aligned
        let (left, not_left) = edge_buf.array.split_at(2 * MAX_TX_SIZE);
        let (_, above) = not_left.split_at(1);

        let above_slice = &above[..B::W + B::H];
        let left_slice = &left[2 * MAX_TX_SIZE - B::H..];

        match mode {
            PredictionMode::DC_PRED => (match variant {
                PredictionVariant::None => B::pred_dc_128,
                PredictionVariant::Left => B::pred_dc_left,
                PredictionVariant::Top => B::pred_dc_top,
                PredictionVariant::Both => B::pred_dc,
            })(dst, above_slice, left_slice, bit_depth),
            _ => unimplemented!(),
        }
    }

    pub trait Intra<T>: Dim
    where
        T: Pixel,
    {
        fn pred_dc(output: &mut PlaneRegionMut<'_, T>, above: &[T], left: &[T], _bit_depth: usize) {
            let edges = left[..Self::H].iter().chain(above[..Self::W].iter());
            let len = (Self::W + Self::H) as u32;
            let avg = (edges.fold(0u32, |acc, &v| {
                let v: u32 = v.into();
                v + acc
            }) + (len >> 1))
                / len;
            let avg = T::cast_from(avg);

            for line in output.rows_iter_mut().take(Self::H) {
                for v in &mut line[..Self::W] {
                    *v = avg;
                }
            }
        }

        fn pred_dc_128(
            output: &mut PlaneRegionMut<'_, T>,
            _above: &[T],
            _left: &[T],
            bit_depth: usize,
        ) {
            let v = T::cast_from(128u32 << (bit_depth - 8));
            for y in 0..Self::H {
                for x in 0..Self::W {
                    output[y][x] = v;
                }
            }
        }

        fn pred_dc_left(
            output: &mut PlaneRegionMut<'_, T>,
            _above: &[T],
            left: &[T],
            _bit_depth: usize,
        ) {
            let sum = left[..Self::H].iter().fold(0u32, |acc, &v| {
                let v: u32 = v.into();
                v + acc
            });
            let avg = T::cast_from((sum + (Self::H >> 1) as u32) / Self::H as u32);
            for line in output.rows_iter_mut().take(Self::H) {
                line[..Self::W].iter_mut().for_each(|v| *v = avg);
            }
        }

        fn pred_dc_top(
            output: &mut PlaneRegionMut<'_, T>,
            above: &[T],
            _left: &[T],
            _bit_depth: usize,
        ) {
            let sum = above[..Self::W].iter().fold(0u32, |acc, &v| {
                let v: u32 = v.into();
                v + acc
            });
            let avg = T::cast_from((sum + (Self::W >> 1) as u32) / Self::W as u32);
            for line in output.rows_iter_mut().take(Self::H) {
                line[..Self::W].iter_mut().for_each(|v| *v = avg);
            }
        }
    }
}
