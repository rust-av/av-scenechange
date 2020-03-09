use crate::cost::{get_sad, get_satd};
use crate::frame::*;
use crate::pred::PredictionMode;
use crate::refs::RefType::NONE_FRAME;
use crate::refs::{
    FrameMotionVectors, MotionVector, RefType, ReferenceFrame, ALL_INTER_REFS, REF_FRAMES,
};
use arrayvec::ArrayVec;
use std::convert::identity;
use std::sync::Arc;
use v_frame::pixel::Pixel;
use v_frame::plane::{Plane, PlaneConfig, PlaneOffset};

#[inline(always)]
pub(crate) fn build_coarse_pmvs<T: Pixel>(
    fi: &FrameInvariants<T>,
    fs: &FrameState<T>,
) -> Vec<[Option<MotionVector>; REF_FRAMES]> {
    if fi.mi_width >= 16 && fi.mi_height >= 16 {
        let mut frame_pmvs = Vec::with_capacity(fi.sb_width * fi.sb_height);
        for sby in 0..fi.sb_height {
            for sbx in 0..fi.sb_width {
                let sbo = PlaneSuperBlockOffset(SuperBlockOffset { x: sbx, y: sby });
                let bo = sbo.block_offset(0, 0);
                let mut pmvs: [Option<MotionVector>; REF_FRAMES] = [None; REF_FRAMES];
                let r = fi.ref_frames[RefType::LAST_FRAME.to_index()] as usize;
                if pmvs[r].is_none() {
                    pmvs[r] = estimate_motion_ss4(fi, fs, BlockSize::BLOCK_64X64, r, bo);
                }
                frame_pmvs.push(pmvs);
            }
        }
        frame_pmvs
    } else {
        // the block use for motion estimation would be smaller than the whole image
        // dynamic allocation: once per frame
        vec![[None; REF_FRAMES]; fi.sb_width * fi.sb_height]
    }
}

pub(crate) fn build_half_res_pmvs<T: Pixel>(
    fi: &FrameInvariants<T>,
    fs: &mut FrameState<T>,
    sbo: PlaneSuperBlockOffset,
    frame_pmvs: &[[Option<MotionVector>; REF_FRAMES]],
) -> BlockPmv {
    let estimate_motion_ss2 = DiamondSearch::estimate_motion_ss2;

    let PlaneSuperBlockOffset(SuperBlockOffset { x: sbx, y: sby }) = sbo;
    let mut pmvs: BlockPmv = [[None; REF_FRAMES]; 5];

    // The pmvs array stores 5 motion vectors in the following order:
    //
    //       64×64
    // ┌───────┬───────┐
    // │       │       │
    // │   1   │   2   │
    // │       ╵       │
    // ├────── 0 ──────┤
    // │       ╷       │
    // │   3   │   4   │
    // │       │       │
    // └───────┴───────┘
    //
    // That is, 0 is the motion vector for the whole 64×64 block, obtained from
    // the quarter-resolution search, and 1 through 4 are the motion vectors for
    // the 32×32 blocks, obtained below from the half-resolution search.
    //
    // Each of the four half-resolution searches uses three quarter-resolution
    // candidates: one from the current 64×64 block and two from the two
    // immediately adjacent 64×64 blocks.
    //
    //          ┌───────┐
    //          │       │
    //          │   n   │
    //          │       │
    //          └───────┘
    // ┌───────┐┌───┬───┐┌───────┐
    // │       ││ 1 ╵ 2 ││       │
    // │   w   │├── 0 ──┤│   e   │
    // │       ││ 3 ╷ 4 ││       │
    // └───────┘└───┴───┘└───────┘
    //          ┌───────┐
    //          │       │
    //          │   s   │
    //          │       │
    //          └───────┘

    if fi.mi_width >= 8 && fi.mi_height >= 8 {
        for &i in ALL_INTER_REFS.iter() {
            let r = fi.ref_frames[i.to_index()] as usize;
            if pmvs[0][r].is_none() {
                pmvs[0][r] = frame_pmvs[sby * fi.sb_width + sbx][r];
                if let Some(pmv) = pmvs[0][r] {
                    let pmv_w = if sbx > 0 {
                        frame_pmvs[sby * fi.sb_width + sbx - 1][r]
                    } else {
                        None
                    };
                    let pmv_e = if sbx < fi.sb_width - 1 {
                        frame_pmvs[sby * fi.sb_width + sbx + 1][r]
                    } else {
                        None
                    };
                    let pmv_n = if sby > 0 {
                        frame_pmvs[sby * fi.sb_width + sbx - fi.sb_width][r]
                    } else {
                        None
                    };
                    let pmv_s = if sby < fi.sb_height - 1 {
                        frame_pmvs[sby * fi.sb_width + sbx + fi.sb_width][r]
                    } else {
                        None
                    };

                    pmvs[1][r] = estimate_motion_ss2(
                        fi,
                        fs,
                        BlockSize::BLOCK_32X32,
                        sbo.block_offset(0, 0),
                        &[Some(pmv), pmv_w, pmv_n],
                        i,
                    );
                    pmvs[2][r] = estimate_motion_ss2(
                        fi,
                        fs,
                        BlockSize::BLOCK_32X32,
                        sbo.block_offset(8, 0),
                        &[Some(pmv), pmv_e, pmv_n],
                        i,
                    );
                    pmvs[3][r] = estimate_motion_ss2(
                        fi,
                        fs,
                        BlockSize::BLOCK_32X32,
                        sbo.block_offset(0, 8),
                        &[Some(pmv), pmv_w, pmv_s],
                        i,
                    );
                    pmvs[4][r] = estimate_motion_ss2(
                        fi,
                        fs,
                        BlockSize::BLOCK_32X32,
                        sbo.block_offset(8, 8),
                        &[Some(pmv), pmv_e, pmv_s],
                        i,
                    );

                    if let Some(mv) = pmvs[1][r] {
                        save_block_motion(
                            fs,
                            fi,
                            BlockSize::BLOCK_32X32,
                            sbo.block_offset(0, 0),
                            i.to_index(),
                            mv,
                        );
                    }
                    if let Some(mv) = pmvs[2][r] {
                        save_block_motion(
                            fs,
                            fi,
                            BlockSize::BLOCK_32X32,
                            sbo.block_offset(8, 0),
                            i.to_index(),
                            mv,
                        );
                    }
                    if let Some(mv) = pmvs[3][r] {
                        save_block_motion(
                            fs,
                            fi,
                            BlockSize::BLOCK_32X32,
                            sbo.block_offset(0, 8),
                            i.to_index(),
                            mv,
                        );
                    }
                    if let Some(mv) = pmvs[4][r] {
                        save_block_motion(
                            fs,
                            fi,
                            BlockSize::BLOCK_32X32,
                            sbo.block_offset(8, 8),
                            i.to_index(),
                            mv,
                        );
                    }
                }
            }
        }
    }

    pmvs
}

pub(crate) fn build_full_res_pmvs<T: Pixel>(
    fi: &FrameInvariants<T>,
    fs: &mut FrameState<T>,
    sbo: PlaneSuperBlockOffset,
    half_res_pmvs: &[[[Option<MotionVector>; REF_FRAMES]; 5]],
) {
    let estimate_motion = DiamondSearch::estimate_motion;

    let PlaneSuperBlockOffset(SuperBlockOffset { x: sbx, y: sby }) = sbo;
    let mut pmvs: [Option<MotionVector>; REF_FRAMES] = [None; REF_FRAMES];
    let half_res_pmvs_this_block = half_res_pmvs[sby * fi.sb_width + sbx];

    if fi.mi_width >= 8 && fi.mi_height >= 8 {
        for &i in ALL_INTER_REFS.iter() {
            let r = fi.ref_frames[i.to_index()] as usize;
            if pmvs[r].is_none() {
                pmvs[r] = half_res_pmvs_this_block[0][r];
                if let Some(pmv) = pmvs[r] {
                    let pmvs_w = if sbx > 0 {
                        half_res_pmvs[sby * fi.sb_width + sbx - 1]
                    } else {
                        [[None; REF_FRAMES]; 5]
                    };
                    let pmvs_e = if sbx < fi.sb_width - 1 {
                        half_res_pmvs[sby * fi.sb_width + sbx + 1]
                    } else {
                        [[None; REF_FRAMES]; 5]
                    };
                    let pmvs_n = if sby > 0 {
                        half_res_pmvs[sby * fi.sb_width + sbx - fi.sb_width]
                    } else {
                        [[None; REF_FRAMES]; 5]
                    };
                    let pmvs_s = if sby < fi.sb_height - 1 {
                        half_res_pmvs[sby * fi.sb_width + sbx + fi.sb_width]
                    } else {
                        [[None; REF_FRAMES]; 5]
                    };

                    for y in 0..4 {
                        for x in 0..4 {
                            let bo = sbo.block_offset(x * 4, y * 4);

                            // We start from half_res_pmvs which include five motion vectors
                            // for a 64×64 block, as described in build_half_res_pmvs. In
                            // this loop we go one level down and search motion vectors for
                            // 16×16 blocks using the full-resolution frames:
                            //
                            //               64×64
                            // ┏━━━━━━━┯━━━━━━━┳━━━━━━━┯━━━━━━━┓
                            // ┃       │       ┃       │       ┃
                            // ┃       │       ┃       │       ┃
                            // ┃       ╵       ┃       ╵       ┃
                            // ┠────── 1 ──────╂────── 2 ──────┨
                            // ┃       ╷       ┃       ╷       ┃
                            // ┃       │       ┃       │       ┃
                            // ┃       │       ╹       │       ┃
                            // ┣━━━━━━━┿━━━━━━ 0 ━━━━━━┿━━━━━━━┫
                            // ┃       │       ╻       │       ┃
                            // ┃       │       ┃       │       ┃
                            // ┃       ╵       ┃       ╵       ┃
                            // ┠────── 3 ──────╂────── 4 ──────┨
                            // ┃       ╷       ┃       ╷       ┃
                            // ┃       │       ┃       │       ┃
                            // ┃       │       ┃       │       ┃
                            // ┗━━━━━━━┷━━━━━━━┻━━━━━━━┷━━━━━━━┛
                            //
                            // Each search receives all covering and adjacent motion vectors
                            // as candidates. Additionally, the middle two rows of blocks
                            // also receive the 32×32 motion vectors from neighboring 64×64
                            // blocks, even though not directly adjacent; same with middle
                            // two columns.
                            let covering_half_res = match (x, y) {
                                (0..=1, 0..=1) => (half_res_pmvs_this_block[1][r]),
                                (2..=3, 0..=1) => (half_res_pmvs_this_block[2][r]),
                                (0..=1, 2..=3) => (half_res_pmvs_this_block[3][r]),
                                (2..=3, 2..=3) => (half_res_pmvs_this_block[4][r]),
                                _ => unreachable!(),
                            };

                            let (vertical_candidate_1, vertical_candidate_2) = match (x, y) {
                                (0..=1, 0) => (pmvs_n[0][r], pmvs_n[3][r]),
                                (2..=3, 0) => (pmvs_n[0][r], pmvs_n[4][r]),
                                (0..=1, 1) => (pmvs_n[3][r], half_res_pmvs_this_block[3][r]),
                                (2..=3, 1) => (pmvs_n[4][r], half_res_pmvs_this_block[4][r]),
                                (0..=1, 2) => (pmvs_s[1][r], half_res_pmvs_this_block[1][r]),
                                (2..=3, 2) => (pmvs_s[2][r], half_res_pmvs_this_block[2][r]),
                                (0..=1, 3) => (pmvs_s[0][r], pmvs_s[1][r]),
                                (2..=3, 3) => (pmvs_s[0][r], pmvs_s[2][r]),
                                _ => unreachable!(),
                            };

                            let (horizontal_candidate_1, horizontal_candidate_2) = match (x, y) {
                                (0, 0..=1) => (pmvs_w[0][r], pmvs_w[2][r]),
                                (0, 2..=3) => (pmvs_w[0][r], pmvs_w[4][r]),
                                (1, 0..=1) => (pmvs_w[2][r], half_res_pmvs_this_block[2][r]),
                                (1, 2..=3) => (pmvs_w[4][r], half_res_pmvs_this_block[4][r]),
                                (2, 0..=1) => (pmvs_e[1][r], half_res_pmvs_this_block[1][r]),
                                (2, 2..=3) => (pmvs_e[3][r], half_res_pmvs_this_block[3][r]),
                                (3, 0..=1) => (pmvs_e[0][r], pmvs_e[2][r]),
                                (3, 2..=3) => (pmvs_e[0][r], pmvs_e[4][r]),
                                _ => unreachable!(),
                            };

                            if let Some(mv) = estimate_motion(
                                fi,
                                fs,
                                BlockSize::BLOCK_16X16,
                                bo,
                                &[
                                    Some(pmv),
                                    covering_half_res,
                                    vertical_candidate_1,
                                    vertical_candidate_2,
                                    horizontal_candidate_1,
                                    horizontal_candidate_2,
                                ],
                                i,
                            ) {
                                save_block_motion(
                                    fs,
                                    fi,
                                    BlockSize::BLOCK_16X16,
                                    bo,
                                    i.to_index(),
                                    mv,
                                );
                            }
                        }
                    }
                }
            }
        }
    }
}

type BlockPmv = [[Option<MotionVector>; REF_FRAMES]; 5];

fn estimate_motion_ss4<T: Pixel>(
    fi: &FrameInvariants<T>,
    fs: &FrameState<T>,
    bsize: BlockSize,
    ref_idx: usize,
    bo: PlaneBlockOffset,
) -> Option<MotionVector> {
    if let Some(ref rec) = fi.rec_buffer.frames[ref_idx] {
        let blk_w = bsize.width();
        let blk_h = bsize.height();
        let po = PlaneOffset {
            x: (bo.0.x as isize) << BLOCK_TO_PLANE_SHIFT >> 2,
            y: (bo.0.y as isize) << BLOCK_TO_PLANE_SHIFT >> 2,
        };

        let range_x = 192 * fi.me_range_scale as isize;
        let range_y = 64 * fi.me_range_scale as isize;
        let (mvx_min, mvx_max, mvy_min, mvy_max) =
            get_mv_range(fi.w_in_b, fi.h_in_b, bo, blk_w, blk_h);
        let x_lo = po.x + (((-range_x).max(mvx_min / 8)) >> 2);
        let x_hi = po.x + (((range_x).min(mvx_max / 8)) >> 2);
        let y_lo = po.y + (((-range_y).max(mvy_min / 8)) >> 2);
        let y_hi = po.y + (((range_y).min(mvy_max / 8)) >> 2);

        let mut lowest_cost = std::u64::MAX;
        let mut best_mv = MotionVector::default();

        // Divide by 16 to account for subsampling, 0.125 is a fudge factor
        let lambda = (fi.me_lambda * 256.0 / 16.0 * 0.125) as u32;

        full_search(
            fi,
            x_lo,
            x_hi,
            y_lo,
            y_hi,
            BlockSize::from_width_and_height(blk_w >> 2, blk_h >> 2),
            &fs.input_qres,
            &rec.input_qres,
            &mut best_mv,
            &mut lowest_cost,
            po,
            1,
            lambda,
            [MotionVector::default(); 2],
            fi.allow_high_precision_mv,
        );

        Some(MotionVector {
            row: best_mv.row * 4,
            col: best_mv.col * 4,
        })
    } else {
        None
    }
}

fn full_search<T: Pixel>(
    fi: &FrameInvariants<T>,
    x_lo: isize,
    x_hi: isize,
    y_lo: isize,
    y_hi: isize,
    bsize: BlockSize,
    p_org: &Plane<T>,
    p_ref: &Plane<T>,
    best_mv: &mut MotionVector,
    lowest_cost: &mut u64,
    po: PlaneOffset,
    step: usize,
    lambda: u32,
    pmv: [MotionVector; 2],
    allow_high_precision_mv: bool,
) {
    let bit_depth = fi.bit_depth;
    let blk_w = bsize.width();
    let blk_h = bsize.height();
    let plane_org = p_org.region(Area::StartingAt { x: po.x, y: po.y });
    let search_region = p_ref.region(Area::Rect {
        x: x_lo,
        y: y_lo,
        width: (x_hi - x_lo) as usize + blk_w,
        height: (y_hi - y_lo) as usize + blk_h,
    });

    // Select rectangular regions within search region with vert+horz windows
    for vert_window in search_region.vert_windows(blk_h).step_by(step) {
        for ref_window in vert_window.horz_windows(blk_w).step_by(step) {
            let sad = get_sad(&plane_org, &ref_window, bsize, bit_depth);

            let &Rect { x, y, .. } = ref_window.rect();

            let mv = MotionVector {
                row: 8 * (y as i16 - po.y as i16),
                col: 8 * (x as i16 - po.x as i16),
            };

            let rate1 = get_mv_rate(mv, pmv[0], allow_high_precision_mv);
            let rate2 = get_mv_rate(mv, pmv[1], allow_high_precision_mv);
            let rate = rate1.min(rate2 + 1);
            let cost = 256 * sad as u64 + rate as u64 * lambda as u64;

            if cost < *lowest_cost {
                *lowest_cost = cost;
                *best_mv = mv;
            }
        }
    }
}

#[inline(always)]
fn get_mv_rate(a: MotionVector, b: MotionVector, allow_high_precision_mv: bool) -> u32 {
    #[inline(always)]
    fn diff_to_rate(diff: i16, allow_high_precision_mv: bool) -> u32 {
        let d = if allow_high_precision_mv {
            diff
        } else {
            diff >> 1
        };
        if d == 0 {
            0
        } else {
            2 * (16 - d.abs().leading_zeros())
        }
    }

    diff_to_rate(a.row - b.row, allow_high_precision_mv)
        + diff_to_rate(a.col - b.col, allow_high_precision_mv)
}

const fn get_mv_range(
    w_in_b: usize,
    h_in_b: usize,
    bo: PlaneBlockOffset,
    blk_w: usize,
    blk_h: usize,
) -> (isize, isize, isize, isize) {
    let border_w = 128 + blk_w as isize * 8;
    let border_h = 128 + blk_h as isize * 8;
    let mvx_min = -(bo.0.x as isize) * (8 * MI_SIZE) as isize - border_w;
    let mvx_max = (w_in_b - bo.0.x - blk_w / MI_SIZE) as isize * (8 * MI_SIZE) as isize + border_w;
    let mvy_min = -(bo.0.y as isize) * (8 * MI_SIZE) as isize - border_h;
    let mvy_max = (h_in_b - bo.0.y - blk_h / MI_SIZE) as isize * (8 * MI_SIZE) as isize + border_h;

    (mvx_min, mvx_max, mvy_min, mvy_max)
}

trait MotionEstimation {
    fn full_pixel_me<T: Pixel>(
        fi: &FrameInvariants<T>,
        fs: &FrameState<T>,
        rec: &ReferenceFrame<T>,
        plane_bo: PlaneBlockOffset,
        lambda: u32,
        cmvs: ArrayVec<[MotionVector; 7]>,
        pmv: [MotionVector; 2],
        mvx_min: isize,
        mvx_max: isize,
        mvy_min: isize,
        mvy_max: isize,
        bsize: BlockSize,
        best_mv: &mut MotionVector,
        lowest_cost: &mut u64,
        ref_frame: RefType,
    );

    fn estimate_motion_ss2<T: Pixel>(
        fi: &FrameInvariants<T>,
        fs: &FrameState<T>,
        bsize: BlockSize,
        plane_bo: PlaneBlockOffset,
        pmvs: &[Option<MotionVector>; 3],
        ref_frame: RefType,
    ) -> Option<MotionVector> {
        if let Some(ref rec) = fi.rec_buffer.frames[fi.ref_frames[ref_frame.to_index()] as usize] {
            let blk_w = bsize.width();
            let blk_h = bsize.height();
            let (mvx_min, mvx_max, mvy_min, mvy_max) =
                get_mv_range(fi.w_in_b, fi.h_in_b, plane_bo, blk_w, blk_h);

            let global_mv = [MotionVector { row: 0, col: 0 }; 2];

            let mut lowest_cost = std::u64::MAX;
            let mut best_mv = MotionVector::default();

            // Divide by 4 to account for subsampling, 0.125 is a fudge factor
            let lambda = (fi.me_lambda * 256.0 / 4.0 * 0.125) as u32;

            Self::me_ss2(
                fi,
                fs,
                pmvs,
                plane_bo,
                rec,
                global_mv,
                lambda,
                mvx_min,
                mvx_max,
                mvy_min,
                mvy_max,
                bsize,
                &mut best_mv,
                &mut lowest_cost,
                ref_frame,
            );

            Some(MotionVector {
                row: best_mv.row * 2,
                col: best_mv.col * 2,
            })
        } else {
            None
        }
    }

    fn me_ss2<T: Pixel>(
        fi: &FrameInvariants<T>,
        fs: &FrameState<T>,
        pmvs: &[Option<MotionVector>; 3],
        plane_bo: PlaneBlockOffset,
        rec: &ReferenceFrame<T>,
        global_mv: [MotionVector; 2],
        lambda: u32,
        mvx_min: isize,
        mvx_max: isize,
        mvy_min: isize,
        mvy_max: isize,
        bsize: BlockSize,
        best_mv: &mut MotionVector,
        lowest_cost: &mut u64,
        ref_frame: RefType,
    );

    fn estimate_motion<T: Pixel>(
        fi: &FrameInvariants<T>,
        fs: &FrameState<T>,
        bsize: BlockSize,
        plane_bo: PlaneBlockOffset,
        pmvs: &[Option<MotionVector>],
        ref_frame: RefType,
    ) -> Option<MotionVector> {
        debug_assert!(pmvs.len() <= 7);

        if let Some(ref rec) = fi.rec_buffer.frames[fi.ref_frames[ref_frame.to_index()] as usize] {
            let blk_w = bsize.width();
            let blk_h = bsize.height();
            let (mvx_min, mvx_max, mvy_min, mvy_max) =
                get_mv_range(fi.w_in_b, fi.h_in_b, plane_bo, blk_w, blk_h);

            let global_mv = [MotionVector { row: 0, col: 0 }; 2];

            let mut lowest_cost = std::u64::MAX;
            let mut best_mv = MotionVector::default();

            // 0.5 is a fudge factor
            let lambda = (fi.me_lambda * 256.0 * 0.5) as u32;

            Self::full_pixel_me(
                fi,
                fs,
                rec,
                plane_bo,
                lambda,
                pmvs.iter().cloned().filter_map(identity).collect(),
                global_mv,
                mvx_min,
                mvx_max,
                mvy_min,
                mvy_max,
                bsize,
                &mut best_mv,
                &mut lowest_cost,
                ref_frame,
            );

            Some(MotionVector {
                row: best_mv.row,
                col: best_mv.col,
            })
        } else {
            None
        }
    }
}

struct DiamondSearch {}

impl MotionEstimation for DiamondSearch {
    fn full_pixel_me<T: Pixel>(
        fi: &FrameInvariants<T>,
        fs: &FrameState<T>,
        rec: &ReferenceFrame<T>,
        plane_bo: PlaneBlockOffset,
        lambda: u32,
        cmvs: ArrayVec<[MotionVector; 7]>,
        pmv: [MotionVector; 2],
        mvx_min: isize,
        mvx_max: isize,
        mvy_min: isize,
        mvy_max: isize,
        bsize: BlockSize,
        best_mv: &mut MotionVector,
        lowest_cost: &mut u64,
        ref_frame: RefType,
    ) {
        let mvs = &fs.frame_mvs[ref_frame.to_index()];
        let frame_ref = fi.rec_buffer.frames[fi.ref_frames[0] as usize]
            .as_ref()
            .map(Arc::as_ref);
        let predictors =
            get_subset_predictors(plane_bo, cmvs, mvs, frame_ref, ref_frame.to_index());

        diamond_me_search(
            fi,
            plane_bo.to_luma_plane_offset(),
            &fs.input.planes[0],
            &rec.frame.planes[0],
            &predictors,
            fi.bit_depth,
            pmv,
            lambda,
            mvx_min,
            mvx_max,
            mvy_min,
            mvy_max,
            bsize,
            false,
            best_mv,
            lowest_cost,
            false,
            ref_frame,
        );
    }

    fn me_ss2<T: Pixel>(
        fi: &FrameInvariants<T>,
        fs: &FrameState<T>,
        pmvs: &[Option<MotionVector>; 3],
        plane_bo: PlaneBlockOffset,
        rec: &ReferenceFrame<T>,
        global_mv: [MotionVector; 2],
        lambda: u32,
        mvx_min: isize,
        mvx_max: isize,
        mvy_min: isize,
        mvy_max: isize,
        bsize: BlockSize,
        best_mv: &mut MotionVector,
        lowest_cost: &mut u64,
        ref_frame: RefType,
    ) {
        let frame_po = PlaneOffset {
            x: (plane_bo.0.x as isize) << BLOCK_TO_PLANE_SHIFT >> 1,
            y: (plane_bo.0.y as isize) << BLOCK_TO_PLANE_SHIFT >> 1,
        };

        let mvs = &fs.frame_mvs[ref_frame.to_index()];
        let frame_ref = fi.rec_buffer.frames[fi.ref_frames[0] as usize]
            .as_ref()
            .map(Arc::as_ref);

        let mut predictors = get_subset_predictors::<T>(
            plane_bo,
            pmvs.iter().cloned().filter_map(identity).collect(),
            &mvs,
            frame_ref,
            ref_frame.to_index(),
        );

        for predictor in &mut predictors {
            predictor.row >>= 1;
            predictor.col >>= 1;
        }

        diamond_me_search(
            fi,
            frame_po,
            &fs.input_hres,
            &rec.input_hres,
            &predictors,
            fi.bit_depth,
            global_mv,
            lambda,
            mvx_min >> 1,
            mvx_max >> 1,
            mvy_min >> 1,
            mvy_max >> 1,
            BlockSize::from_width_and_height(bsize.width() >> 1, bsize.height() >> 1),
            false,
            best_mv,
            lowest_cost,
            false,
            ref_frame,
        );
    }
}

pub fn get_subset_predictors<T: Pixel>(
    frame_bo: PlaneBlockOffset,
    cmvs: ArrayVec<[MotionVector; 7]>,
    frame_mvs: &FrameMotionVectors,
    frame_ref_opt: Option<&ReferenceFrame<T>>,
    ref_frame_id: usize,
) -> ArrayVec<[MotionVector; 17]> {
    let mut predictors = ArrayVec::<[_; 17]>::new();

    // Zero motion vector
    predictors.push(MotionVector::default());

    // Coarse motion estimation.
    predictors.extend(cmvs.into_iter().map(MotionVector::quantize_to_fullpel));

    // EPZS subset A and B predictors.

    let mut median_preds = ArrayVec::<[_; 3]>::new();
    if frame_bo.0.x > 0 {
        let left = frame_mvs[frame_bo.0.y][frame_bo.0.x - 1];
        median_preds.push(left);
        if !left.is_zero() {
            predictors.push(left);
        }
    }
    if frame_bo.0.y > 0 {
        let top = frame_mvs[frame_bo.0.y - 1][frame_bo.0.x];
        median_preds.push(top);
        if !top.is_zero() {
            predictors.push(top);
        }

        if frame_bo.0.x < frame_mvs.cols - 1 {
            let top_right = frame_mvs[frame_bo.0.y - 1][frame_bo.0.x + 1];
            median_preds.push(top_right);
            if !top_right.is_zero() {
                predictors.push(top_right);
            }
        }
    }

    if !median_preds.is_empty() {
        let mut median_mv = MotionVector::default();
        for mv in median_preds.iter() {
            median_mv = median_mv + *mv;
        }
        median_mv = median_mv / (median_preds.len() as i16);
        let median_mv_quant = median_mv.quantize_to_fullpel();
        if !median_mv_quant.is_zero() {
            predictors.push(median_mv_quant);
        }
    }

    // EPZS subset C predictors.

    if let Some(ref frame_ref) = frame_ref_opt {
        let prev_frame_mvs = &frame_ref.frame_mvs[ref_frame_id];
        if frame_bo.0.x > 0 {
            let left = prev_frame_mvs[frame_bo.0.y][frame_bo.0.x - 1];
            if !left.is_zero() {
                predictors.push(left);
            }
        }
        if frame_bo.0.y > 0 {
            let top = prev_frame_mvs[frame_bo.0.y - 1][frame_bo.0.x];
            if !top.is_zero() {
                predictors.push(top);
            }
        }
        if frame_bo.0.x < prev_frame_mvs.cols - 1 {
            let right = prev_frame_mvs[frame_bo.0.y][frame_bo.0.x + 1];
            if !right.is_zero() {
                predictors.push(right);
            }
        }
        if frame_bo.0.y < prev_frame_mvs.rows - 1 {
            let bottom = prev_frame_mvs[frame_bo.0.y + 1][frame_bo.0.x];
            if !bottom.is_zero() {
                predictors.push(bottom);
            }
        }

        let previous = prev_frame_mvs[frame_bo.0.y][frame_bo.0.x];
        if !previous.is_zero() {
            predictors.push(previous);
        }
    }

    predictors
}

fn save_block_motion<T: Pixel>(
    fs: &mut FrameState<T>,
    fi: &FrameInvariants<T>,
    bsize: BlockSize,
    frame_bo: PlaneBlockOffset,
    ref_frame: usize,
    mv: MotionVector,
) {
    let frame_mvs = &mut Arc::make_mut(&mut fs.frame_mvs)[ref_frame];
    let frame_bo_x_end = (frame_bo.0.x + bsize.width_mi()).min(fi.mi_width);
    let frame_bo_y_end = (frame_bo.0.y + bsize.height_mi()).min(fi.mi_height);
    for mi_y in frame_bo.0.y..frame_bo_y_end {
        for mi_x in frame_bo.0.x..frame_bo_x_end {
            frame_mvs[mi_y][mi_x] = mv;
        }
    }
}

fn diamond_me_search<T: Pixel>(
    fi: &FrameInvariants<T>,
    po: PlaneOffset,
    p_org: &Plane<T>,
    p_ref: &Plane<T>,
    predictors: &[MotionVector],
    bit_depth: usize,
    pmv: [MotionVector; 2],
    lambda: u32,
    mvx_min: isize,
    mvx_max: isize,
    mvy_min: isize,
    mvy_max: isize,
    bsize: BlockSize,
    use_satd: bool,
    center_mv: &mut MotionVector,
    center_mv_cost: &mut u64,
    subpixel: bool,
    ref_frame: RefType,
) {
    use crate::util::AlignedArray;

    let cfg = PlaneConfig::new(
        bsize.width(),
        bsize.height(),
        0,
        0,
        0,
        0,
        std::mem::size_of::<T>(),
    );

    let mut buf: AlignedArray<[T; 128 * 128]> = AlignedArray::uninitialized();

    let diamond_pattern = [(1i16, 0i16), (0, 1), (-1, 0), (0, -1)];
    let (mut diamond_radius, diamond_radius_end, mut tmp_region_opt) = {
        if subpixel {
            let rect = Rect {
                x: 0,
                y: 0,
                width: cfg.width,
                height: cfg.height,
            };

            // Sub-pixel motion estimation
            (
                4i16,
                if fi.allow_high_precision_mv {
                    1i16
                } else {
                    2i16
                },
                Some(PlaneRegionMut::from_slice(&mut buf.array, &cfg, rect)),
            )
        } else {
            // Full pixel motion estimation
            (16i16, 8i16, None)
        }
    };

    get_best_predictor(
        fi,
        po,
        p_org,
        p_ref,
        &predictors,
        bit_depth,
        pmv,
        lambda,
        use_satd,
        mvx_min,
        mvx_max,
        mvy_min,
        mvy_max,
        bsize,
        center_mv,
        center_mv_cost,
        &mut tmp_region_opt,
        ref_frame,
    );

    loop {
        let mut best_diamond_rd_cost = std::u64::MAX;
        let mut best_diamond_mv = MotionVector::default();

        for p in diamond_pattern.iter() {
            let cand_mv = MotionVector {
                row: center_mv.row + diamond_radius * p.0,
                col: center_mv.col + diamond_radius * p.1,
            };

            let rd_cost = get_mv_rd_cost(
                fi,
                po,
                p_org,
                p_ref,
                bit_depth,
                pmv,
                lambda,
                use_satd,
                mvx_min,
                mvx_max,
                mvy_min,
                mvy_max,
                bsize,
                cand_mv,
                tmp_region_opt.as_mut(),
                ref_frame,
            );

            if rd_cost < best_diamond_rd_cost {
                best_diamond_rd_cost = rd_cost;
                best_diamond_mv = cand_mv;
            }
        }

        if *center_mv_cost <= best_diamond_rd_cost {
            if diamond_radius == diamond_radius_end {
                break;
            } else {
                diamond_radius /= 2;
            }
        } else {
            *center_mv = best_diamond_mv;
            *center_mv_cost = best_diamond_rd_cost;
        }
    }

    assert!(*center_mv_cost < std::u64::MAX);
}

fn get_best_predictor<T: Pixel>(
    fi: &FrameInvariants<T>,
    po: PlaneOffset,
    p_org: &Plane<T>,
    p_ref: &Plane<T>,
    predictors: &[MotionVector],
    bit_depth: usize,
    pmv: [MotionVector; 2],
    lambda: u32,
    use_satd: bool,
    mvx_min: isize,
    mvx_max: isize,
    mvy_min: isize,
    mvy_max: isize,
    bsize: BlockSize,
    center_mv: &mut MotionVector,
    center_mv_cost: &mut u64,
    tmp_region_opt: &mut Option<PlaneRegionMut<T>>,
    ref_frame: RefType,
) {
    *center_mv = MotionVector::default();
    *center_mv_cost = std::u64::MAX;

    for &init_mv in predictors.iter() {
        let cost = get_mv_rd_cost(
            fi,
            po,
            p_org,
            p_ref,
            bit_depth,
            pmv,
            lambda,
            use_satd,
            mvx_min,
            mvx_max,
            mvy_min,
            mvy_max,
            bsize,
            init_mv,
            tmp_region_opt.as_mut(),
            ref_frame,
        );

        if cost < *center_mv_cost {
            *center_mv = init_mv;
            *center_mv_cost = cost;
        }
    }
}

fn get_mv_rd_cost<T: Pixel>(
    fi: &FrameInvariants<T>,
    po: PlaneOffset,
    p_org: &Plane<T>,
    p_ref: &Plane<T>,
    bit_depth: usize,
    pmv: [MotionVector; 2],
    lambda: u32,
    use_satd: bool,
    mvx_min: isize,
    mvx_max: isize,
    mvy_min: isize,
    mvy_max: isize,
    bsize: BlockSize,
    cand_mv: MotionVector,
    tmp_region_opt: Option<&mut PlaneRegionMut<T>>,
    ref_frame: RefType,
) -> u64 {
    if (cand_mv.col as isize) < mvx_min || (cand_mv.col as isize) > mvx_max {
        return std::u64::MAX;
    }
    if (cand_mv.row as isize) < mvy_min || (cand_mv.row as isize) > mvy_max {
        return std::u64::MAX;
    }

    let plane_org = p_org.region(Area::StartingAt { x: po.x, y: po.y });

    if let Some(region) = tmp_region_opt {
        PredictionMode::NEWMV.predict_inter(
            fi,
            0,
            po,
            region,
            bsize.width(),
            bsize.height(),
            [ref_frame, NONE_FRAME],
            [cand_mv, MotionVector { row: 0, col: 0 }],
        );
        let plane_ref = region.as_const();
        compute_mv_rd_cost(
            fi, pmv, lambda, use_satd, bit_depth, bsize, cand_mv, &plane_org, &plane_ref,
        )
    } else {
        // Full pixel motion vector
        let plane_ref = p_ref.region(Area::StartingAt {
            x: po.x + (cand_mv.col / 8) as isize,
            y: po.y + (cand_mv.row / 8) as isize,
        });
        compute_mv_rd_cost(
            fi, pmv, lambda, use_satd, bit_depth, bsize, cand_mv, &plane_org, &plane_ref,
        )
    }
}

fn compute_mv_rd_cost<T: Pixel>(
    fi: &FrameInvariants<T>,
    pmv: [MotionVector; 2],
    lambda: u32,
    use_satd: bool,
    bit_depth: usize,
    bsize: BlockSize,
    cand_mv: MotionVector,
    plane_org: &PlaneRegion<'_, T>,
    plane_ref: &PlaneRegion<'_, T>,
) -> u64 {
    let sad = if use_satd {
        get_satd(&plane_org, &plane_ref, bsize, bit_depth)
    } else {
        get_sad(&plane_org, &plane_ref, bsize, bit_depth)
    };

    let rate1 = get_mv_rate(cand_mv, pmv[0], fi.allow_high_precision_mv);
    let rate2 = get_mv_rate(cand_mv, pmv[1], fi.allow_high_precision_mv);
    let rate = rate1.min(rate2 + 1);

    256 * sad as u64 + rate as u64 * lambda as u64
}
