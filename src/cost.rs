use crate::frame::*;
use crate::me::*;
use crate::pred::{get_intra_edges, PredictionMode};
use v_frame::frame::Frame;
use v_frame::math::msb;
use v_frame::pixel::{CastFromPrimitive, ChromaSampling, Pixel, PixelType};
use v_frame::plane::PlaneOffset;

#[cfg(not(target_arch = "x86_64"))]
pub use native::*;
#[cfg(target_arch = "x86_64")]
pub use x86::*;

/// Size of blocks for the importance computation, in pixels.
const IMPORTANCE_BLOCK_SIZE: usize = 8;
const IMP_BLOCK_MV_UNITS_PER_PIXEL: i64 = 8;
const IMP_BLOCK_SIZE_IN_MV_UNITS: i64 = IMPORTANCE_BLOCK_SIZE as i64 * IMP_BLOCK_MV_UNITS_PER_PIXEL;

pub(crate) fn estimate_intra_costs<T: Pixel>(frame: &Frame<T>, bit_depth: usize) -> Box<[u32]> {
    let plane = &frame.planes[0];
    let mut plane_after_prediction = frame.planes[0].clone();

    let bsize = BlockSize::from_width_and_height(IMPORTANCE_BLOCK_SIZE, IMPORTANCE_BLOCK_SIZE);
    let tx_size = bsize.tx_size();

    let h_in_imp_b = plane.cfg.height / IMPORTANCE_BLOCK_SIZE;
    let w_in_imp_b = plane.cfg.width / IMPORTANCE_BLOCK_SIZE;
    let mut intra_costs = Vec::with_capacity(h_in_imp_b * w_in_imp_b);

    for y in 0..h_in_imp_b {
        for x in 0..w_in_imp_b {
            let plane_org = plane.region(Area::Rect {
                x: (x * IMPORTANCE_BLOCK_SIZE) as isize,
                y: (y * IMPORTANCE_BLOCK_SIZE) as isize,
                width: IMPORTANCE_BLOCK_SIZE,
                height: IMPORTANCE_BLOCK_SIZE,
            });

            // TODO: other intra prediction modes.
            let edge_buf = get_intra_edges(
                &plane.as_region(),
                PlaneBlockOffset(BlockOffset { x, y }),
                0,
                0,
                bsize,
                PlaneOffset {
                    x: (x * IMPORTANCE_BLOCK_SIZE) as isize,
                    y: (y * IMPORTANCE_BLOCK_SIZE) as isize,
                },
                TxSize::TX_8X8,
                bit_depth,
                Some(PredictionMode::DC_PRED),
            );

            let mut plane_after_prediction_region = plane_after_prediction.region_mut(Area::Rect {
                x: (x * IMPORTANCE_BLOCK_SIZE) as isize,
                y: (y * IMPORTANCE_BLOCK_SIZE) as isize,
                width: IMPORTANCE_BLOCK_SIZE,
                height: IMPORTANCE_BLOCK_SIZE,
            });

            PredictionMode::DC_PRED.predict_intra(
                Rect {
                    x: (x * IMPORTANCE_BLOCK_SIZE) as isize,
                    y: (y * IMPORTANCE_BLOCK_SIZE) as isize,
                    width: IMPORTANCE_BLOCK_SIZE,
                    height: IMPORTANCE_BLOCK_SIZE,
                },
                &mut plane_after_prediction_region,
                tx_size,
                bit_depth,
                &edge_buf,
            );

            let plane_after_prediction_region = plane_after_prediction.region(Area::Rect {
                x: (x * IMPORTANCE_BLOCK_SIZE) as isize,
                y: (y * IMPORTANCE_BLOCK_SIZE) as isize,
                width: IMPORTANCE_BLOCK_SIZE,
                height: IMPORTANCE_BLOCK_SIZE,
            });

            let intra_cost = get_satd(&plane_org, &plane_after_prediction_region, bsize, bit_depth);

            intra_costs.push(intra_cost);
        }
    }

    intra_costs.into_boxed_slice()
}

pub(crate) fn estimate_inter_costs<T: Pixel>(
    frame: &Frame<T>,
    ref_frame: &Frame<T>,
    bit_depth: usize,
    chroma_sampling: ChromaSampling,
) -> Box<[u32]> {
    let plane_org = &frame.planes[0];
    let last_fi = FrameInvariants::new_key_frame(
        plane_org.cfg.width,
        plane_org.cfg.height,
        bit_depth,
        chroma_sampling,
        0,
    );
    let mut fi = FrameInvariants::new_inter_frame(&last_fi);

    // Compute the motion vectors.
    let mut fs = FrameState::new_with_frame(&fi, frame);
    compute_motion_vectors(&mut fi, &mut fs);

    // Estimate inter costs
    let plane_ref = &ref_frame.planes[0];
    let h_in_imp_b = plane_org.cfg.height / IMPORTANCE_BLOCK_SIZE;
    let w_in_imp_b = plane_org.cfg.width / IMPORTANCE_BLOCK_SIZE;
    let mut inter_costs = Vec::with_capacity(h_in_imp_b * w_in_imp_b);
    let mvs = &fs.frame_mvs[0];
    let bsize = BlockSize::from_width_and_height(IMPORTANCE_BLOCK_SIZE, IMPORTANCE_BLOCK_SIZE);
    (0..h_in_imp_b).for_each(|y| {
        (0..w_in_imp_b).for_each(|x| {
            let mv = mvs[y * 2][x * 2];

            // Coordinates of the top-left corner of the reference block, in MV
            // units.
            let reference_x = x as i64 * IMP_BLOCK_SIZE_IN_MV_UNITS + mv.col as i64;
            let reference_y = y as i64 * IMP_BLOCK_SIZE_IN_MV_UNITS + mv.row as i64;

            let region_org = plane_org.region(Area::Rect {
                x: (x * IMPORTANCE_BLOCK_SIZE) as isize,
                y: (y * IMPORTANCE_BLOCK_SIZE) as isize,
                width: IMPORTANCE_BLOCK_SIZE,
                height: IMPORTANCE_BLOCK_SIZE,
            });

            let region_ref = plane_ref.region(Area::Rect {
                x: reference_x as isize / IMP_BLOCK_MV_UNITS_PER_PIXEL as isize,
                y: reference_y as isize / IMP_BLOCK_MV_UNITS_PER_PIXEL as isize,
                width: IMPORTANCE_BLOCK_SIZE,
                height: IMPORTANCE_BLOCK_SIZE,
            });

            inter_costs.push(get_satd(&region_org, &region_ref, bsize, bit_depth));
        });
    });
    inter_costs.into_boxed_slice()
}

fn compute_motion_vectors<T: Pixel>(fi: &mut FrameInvariants<T>, fs: &mut FrameState<T>) {
    // Compute the quarter-resolution motion vectors.
    let frame_pmvs = build_coarse_pmvs(fi, fs);

    // Compute the half-resolution motion vectors.
    let mut half_res_pmvs = Vec::with_capacity(fi.sb_height * fi.sb_width);
    for sby in 0..fi.sb_height {
        for sbx in 0..fi.sb_width {
            let sbo = PlaneSuperBlockOffset(SuperBlockOffset { x: sbx, y: sby });
            half_res_pmvs.push(build_half_res_pmvs(fi, fs, sbo, &frame_pmvs));
        }
    }

    // Compute the full-resolution motion vectors.
    for sby in 0..fi.sb_height {
        for sbx in 0..fi.sb_width {
            let sbo = PlaneSuperBlockOffset(SuperBlockOffset { x: sbx, y: sby });
            build_full_res_pmvs(fi, fs, sbo, &half_res_pmvs);
        }
    }
}

mod native {
    use super::*;

    pub fn get_sad<T: Pixel>(
        plane_org: &PlaneRegion<'_, T>,
        plane_ref: &PlaneRegion<'_, T>,
        bsize: BlockSize,
        _bit_depth: usize,
    ) -> u32 {
        let blk_w = bsize.width();
        let blk_h = bsize.height();

        let mut sum = 0 as u32;

        for (slice_org, slice_ref) in plane_org.rows_iter().take(blk_h).zip(plane_ref.rows_iter()) {
            sum += slice_org
                .iter()
                .take(blk_w)
                .zip(slice_ref)
                .map(|(&a, &b)| (i32::cast_from(a) - i32::cast_from(b)).abs() as u32)
                .sum::<u32>();
        }

        sum
    }

    /// Sum of absolute transformed differences
    /// Use the sum of 4x4 and 8x8 hadamard transforms for the transform. 4x* and
    /// *x4 blocks use 4x4 and all others use 8x8.
    pub fn get_satd<T: Pixel>(
        plane_org: &PlaneRegion<'_, T>,
        plane_ref: &PlaneRegion<'_, T>,
        bsize: BlockSize,
        _bit_depth: usize,
    ) -> u32 {
        let blk_w = bsize.width();
        let blk_h = bsize.height();

        // Size of hadamard transform should be 4x4 or 8x8
        // 4x* and *x4 use 4x4 and all other use 8x8
        let size: usize = blk_w.min(blk_h).min(8);
        let tx2d = if size == 4 { hadamard4x4 } else { hadamard8x8 };

        let mut sum = 0 as u64;

        // Loop over chunks the size of the chosen transform
        for chunk_y in (0..blk_h).step_by(size) {
            for chunk_x in (0..blk_w).step_by(size) {
                let chunk_area: Area = Area::Rect {
                    x: chunk_x as isize,
                    y: chunk_y as isize,
                    width: size,
                    height: size,
                };
                let chunk_org = plane_org.subregion(chunk_area);
                let chunk_ref = plane_ref.subregion(chunk_area);
                let buf: &mut [i32] = &mut [0; 8 * 8][..size * size];

                // Move the difference of the transforms to a buffer
                for (row_diff, (row_org, row_ref)) in buf
                    .chunks_mut(size)
                    .zip(chunk_org.rows_iter().zip(chunk_ref.rows_iter()))
                {
                    for (diff, (a, b)) in
                        row_diff.iter_mut().zip(row_org.iter().zip(row_ref.iter()))
                    {
                        *diff = i32::cast_from(*a) - i32::cast_from(*b);
                    }
                }

                // Perform the hadamard transform on the differences
                tx2d(buf);

                // Sum the absolute values of the transformed differences
                sum += buf.iter().map(|a| a.abs() as u64).sum::<u64>();
            }
        }

        // Normalize the results
        let ln = msb(size as i32) as u64;
        ((sum + (1 << ln >> 1)) >> ln) as u32
    }

    #[inline(always)]
    const fn butterfly(a: i32, b: i32) -> (i32, i32) {
        ((a + b), (a - b))
    }

    #[inline(always)]
    #[allow(clippy::identity_op, clippy::erasing_op)]
    fn hadamard4_1d(data: &mut [i32], n: usize, stride0: usize, stride1: usize) {
        for i in 0..n {
            let sub: &mut [i32] = &mut data[i * stride0..];
            let (a0, a1) = butterfly(sub[0 * stride1], sub[1 * stride1]);
            let (a2, a3) = butterfly(sub[2 * stride1], sub[3 * stride1]);
            let (b0, b2) = butterfly(a0, a2);
            let (b1, b3) = butterfly(a1, a3);
            sub[0 * stride1] = b0;
            sub[1 * stride1] = b1;
            sub[2 * stride1] = b2;
            sub[3 * stride1] = b3;
        }
    }

    #[inline(always)]
    #[allow(clippy::identity_op, clippy::erasing_op)]
    fn hadamard8_1d(data: &mut [i32], n: usize, stride0: usize, stride1: usize) {
        for i in 0..n {
            let sub: &mut [i32] = &mut data[i * stride0..];

            let (a0, a1) = butterfly(sub[0 * stride1], sub[1 * stride1]);
            let (a2, a3) = butterfly(sub[2 * stride1], sub[3 * stride1]);
            let (a4, a5) = butterfly(sub[4 * stride1], sub[5 * stride1]);
            let (a6, a7) = butterfly(sub[6 * stride1], sub[7 * stride1]);

            let (b0, b2) = butterfly(a0, a2);
            let (b1, b3) = butterfly(a1, a3);
            let (b4, b6) = butterfly(a4, a6);
            let (b5, b7) = butterfly(a5, a7);

            let (c0, c4) = butterfly(b0, b4);
            let (c1, c5) = butterfly(b1, b5);
            let (c2, c6) = butterfly(b2, b6);
            let (c3, c7) = butterfly(b3, b7);

            sub[0 * stride1] = c0;
            sub[1 * stride1] = c1;
            sub[2 * stride1] = c2;
            sub[3 * stride1] = c3;
            sub[4 * stride1] = c4;
            sub[5 * stride1] = c5;
            sub[6 * stride1] = c6;
            sub[7 * stride1] = c7;
        }
    }

    #[inline(always)]
    fn hadamard2d(data: &mut [i32], (w, h): (usize, usize)) {
        /*Vertical transform.*/
        let vert_func = if h == 4 { hadamard4_1d } else { hadamard8_1d };
        vert_func(data, w, 1, h);
        /*Horizontal transform.*/
        let horz_func = if w == 4 { hadamard4_1d } else { hadamard8_1d };
        horz_func(data, h, w, 1);
    }

    fn hadamard4x4(data: &mut [i32]) {
        hadamard2d(data, (4, 4));
    }

    fn hadamard8x8(data: &mut [i32]) {
        hadamard2d(data, (8, 8));
    }
}

#[cfg(target_arch = "x86_64")]
mod x86 {
    use super::*;

    macro_rules! declare_asm_dist_fn {
        ($(($name: ident, $T: ident)),+) => (
            $(
                extern { fn $name (
                    src: *const $T, src_stride: isize, dst: *const $T, dst_stride: isize
                ) -> u32; }
            )+
        )
    }

    declare_asm_dist_fn![
        // SAD
        // SSSE3
        (scenechangeasm_sad_4x4_hbd_ssse3, u16),
        (scenechangeasm_sad_16x16_hbd_ssse3, u16),
        // SSE2
        (scenechangeasm_sad4x4_sse2, u8),
        (scenechangeasm_sad4x8_sse2, u8),
        (scenechangeasm_sad4x16_sse2, u8),
        (scenechangeasm_sad8x4_sse2, u8),
        (scenechangeasm_sad8x8_sse2, u8),
        (scenechangeasm_sad8x16_sse2, u8),
        (scenechangeasm_sad8x32_sse2, u8),
        (scenechangeasm_sad16x16_sse2, u8),
        (scenechangeasm_sad32x32_sse2, u8),
        (scenechangeasm_sad64x64_sse2, u8),
        (scenechangeasm_sad128x128_sse2, u8),
        // AVX
        (scenechangeasm_sad16x4_avx2, u8),
        (scenechangeasm_sad16x8_avx2, u8),
        (scenechangeasm_sad16x16_avx2, u8),
        (scenechangeasm_sad16x32_avx2, u8),
        (scenechangeasm_sad16x64_avx2, u8),
        (scenechangeasm_sad32x8_avx2, u8),
        (scenechangeasm_sad32x16_avx2, u8),
        (scenechangeasm_sad32x32_avx2, u8),
        (scenechangeasm_sad32x64_avx2, u8),
        (scenechangeasm_sad64x16_avx2, u8),
        (scenechangeasm_sad64x32_avx2, u8),
        (scenechangeasm_sad64x64_avx2, u8),
        (scenechangeasm_sad64x128_avx2, u8),
        (scenechangeasm_sad128x64_avx2, u8),
        (scenechangeasm_sad128x128_avx2, u8),
        // SATD
        // SSSE3
        (scenechangeasm_satd_8x8_ssse3, u8),
        // SSE4
        (scenechangeasm_satd_4x4_sse4, u8),
        // AVX
        (scenechangeasm_satd_4x4_avx2, u8),
        (scenechangeasm_satd_8x8_avx2, u8),
        (scenechangeasm_satd_16x16_avx2, u8),
        (scenechangeasm_satd_32x32_avx2, u8),
        (scenechangeasm_satd_64x64_avx2, u8),
        (scenechangeasm_satd_128x128_avx2, u8),
        (scenechangeasm_satd_4x8_avx2, u8),
        (scenechangeasm_satd_8x4_avx2, u8),
        (scenechangeasm_satd_8x16_avx2, u8),
        (scenechangeasm_satd_16x8_avx2, u8),
        (scenechangeasm_satd_16x32_avx2, u8),
        (scenechangeasm_satd_32x16_avx2, u8),
        (scenechangeasm_satd_32x64_avx2, u8),
        (scenechangeasm_satd_64x32_avx2, u8),
        (scenechangeasm_satd_64x128_avx2, u8),
        (scenechangeasm_satd_128x64_avx2, u8),
        (scenechangeasm_satd_4x16_avx2, u8),
        (scenechangeasm_satd_16x4_avx2, u8),
        (scenechangeasm_satd_8x32_avx2, u8),
        (scenechangeasm_satd_32x8_avx2, u8),
        (scenechangeasm_satd_16x64_avx2, u8),
        (scenechangeasm_satd_64x16_avx2, u8)
    ];

    #[inline(always)]
    pub fn get_sad<T: Pixel>(
        src: &PlaneRegion<'_, T>,
        dst: &PlaneRegion<'_, T>,
        bsize: BlockSize,
        bit_depth: usize,
    ) -> u32 {
        let call_native = || -> u32 { native::get_sad(dst, src, bsize, bit_depth) };

        match T::type_enum() {
            PixelType::U8 => match get_sad_fns()[to_index(bsize)] {
                Some(func) => unsafe {
                    (func)(
                        src.data_ptr() as *const _,
                        T::to_asm_stride(src.plane_cfg.stride),
                        dst.data_ptr() as *const _,
                        T::to_asm_stride(dst.plane_cfg.stride),
                    )
                },
                None => call_native(),
            },
            PixelType::U16 => match get_sad_fns_hbd()[to_index(bsize)] {
                Some(func) => unsafe {
                    (func)(
                        src.data_ptr() as *const _,
                        T::to_asm_stride(src.plane_cfg.stride),
                        dst.data_ptr() as *const _,
                        T::to_asm_stride(dst.plane_cfg.stride),
                    )
                },
                None => call_native(),
            },
        }
    }

    type SadFn = unsafe extern "C" fn(
        src: *const u8,
        src_stride: isize,
        dst: *const u8,
        dst_stride: isize,
    ) -> u32;

    fn get_sad_fns() -> &'static [Option<SadFn>; DIST_FNS_LENGTH] {
        if is_x86_feature_detected!("avx2") {
            &SAD_FNS_AVX2
        } else if is_x86_feature_detected!("sse2") {
            &SAD_FNS_SSE2
        } else {
            &[None; DIST_FNS_LENGTH]
        }
    }

    static SAD_FNS_SSE2: [Option<SadFn>; DIST_FNS_LENGTH] = {
        let mut out: [Option<SadFn>; DIST_FNS_LENGTH] = [None; DIST_FNS_LENGTH];

        use BlockSize::*;

        out[BLOCK_4X4 as usize] = Some(scenechangeasm_sad4x4_sse2);
        out[BLOCK_4X8 as usize] = Some(scenechangeasm_sad4x8_sse2);
        out[BLOCK_4X16 as usize] = Some(scenechangeasm_sad4x16_sse2);

        out[BLOCK_8X4 as usize] = Some(scenechangeasm_sad8x4_sse2);
        out[BLOCK_8X8 as usize] = Some(scenechangeasm_sad8x8_sse2);
        out[BLOCK_8X16 as usize] = Some(scenechangeasm_sad8x16_sse2);
        out[BLOCK_8X32 as usize] = Some(scenechangeasm_sad8x32_sse2);

        out[BLOCK_16X16 as usize] = Some(scenechangeasm_sad16x16_sse2);
        out[BLOCK_32X32 as usize] = Some(scenechangeasm_sad32x32_sse2);
        out[BLOCK_64X64 as usize] = Some(scenechangeasm_sad64x64_sse2);
        out[BLOCK_128X128 as usize] = Some(scenechangeasm_sad128x128_sse2);

        out
    };

    static SAD_FNS_AVX2: [Option<SadFn>; DIST_FNS_LENGTH] = {
        let mut out: [Option<SadFn>; DIST_FNS_LENGTH] = [None; DIST_FNS_LENGTH];

        use BlockSize::*;

        out[BLOCK_4X4 as usize] = Some(scenechangeasm_sad4x4_sse2);
        out[BLOCK_4X8 as usize] = Some(scenechangeasm_sad4x8_sse2);
        out[BLOCK_4X16 as usize] = Some(scenechangeasm_sad4x16_sse2);

        out[BLOCK_8X4 as usize] = Some(scenechangeasm_sad8x4_sse2);
        out[BLOCK_8X8 as usize] = Some(scenechangeasm_sad8x8_sse2);
        out[BLOCK_8X16 as usize] = Some(scenechangeasm_sad8x16_sse2);
        out[BLOCK_8X32 as usize] = Some(scenechangeasm_sad8x32_sse2);

        out[BLOCK_16X4 as usize] = Some(scenechangeasm_sad16x4_avx2);
        out[BLOCK_16X8 as usize] = Some(scenechangeasm_sad16x8_avx2);
        out[BLOCK_16X16 as usize] = Some(scenechangeasm_sad16x16_avx2);
        out[BLOCK_16X32 as usize] = Some(scenechangeasm_sad16x32_avx2);
        out[BLOCK_16X64 as usize] = Some(scenechangeasm_sad16x64_avx2);

        out[BLOCK_32X8 as usize] = Some(scenechangeasm_sad32x8_avx2);
        out[BLOCK_32X16 as usize] = Some(scenechangeasm_sad32x16_avx2);
        out[BLOCK_32X32 as usize] = Some(scenechangeasm_sad32x32_avx2);
        out[BLOCK_32X64 as usize] = Some(scenechangeasm_sad32x64_avx2);

        out[BLOCK_64X16 as usize] = Some(scenechangeasm_sad64x16_avx2);
        out[BLOCK_64X32 as usize] = Some(scenechangeasm_sad64x32_avx2);
        out[BLOCK_64X64 as usize] = Some(scenechangeasm_sad64x64_avx2);
        out[BLOCK_64X128 as usize] = Some(scenechangeasm_sad64x128_avx2);

        out[BLOCK_128X64 as usize] = Some(scenechangeasm_sad128x64_avx2);
        out[BLOCK_128X128 as usize] = Some(scenechangeasm_sad128x128_avx2);

        out
    };

    type SadHBDFn = unsafe extern "C" fn(
        src: *const u16,
        src_stride: isize,
        dst: *const u16,
        dst_stride: isize,
    ) -> u32;

    fn get_sad_fns_hbd() -> &'static [Option<SadHBDFn>; DIST_FNS_LENGTH] {
        if is_x86_feature_detected!("ssse3") {
            &SAD_HBD_FNS_SSSE3
        } else {
            &[None; DIST_FNS_LENGTH]
        }
    }

    static SAD_HBD_FNS_SSSE3: [Option<SadHBDFn>; DIST_FNS_LENGTH] = {
        let mut out: [Option<SadHBDFn>; DIST_FNS_LENGTH] = [None; DIST_FNS_LENGTH];

        use BlockSize::*;

        out[BLOCK_4X4 as usize] = Some(scenechangeasm_sad_4x4_hbd_ssse3);
        out[BLOCK_16X16 as usize] = Some(scenechangeasm_sad_16x16_hbd_ssse3);

        out
    };

    #[inline(always)]
    pub fn get_satd<T: Pixel>(
        src: &PlaneRegion<'_, T>,
        dst: &PlaneRegion<'_, T>,
        bsize: BlockSize,
        bit_depth: usize,
    ) -> u32 {
        let call_native = || -> u32 { native::get_satd(dst, src, bsize, bit_depth) };

        match T::type_enum() {
            PixelType::U8 => match get_satd_fns()[to_index(bsize)] {
                Some(func) => unsafe {
                    (func)(
                        src.data_ptr() as *const _,
                        T::to_asm_stride(src.plane_cfg.stride),
                        dst.data_ptr() as *const _,
                        T::to_asm_stride(dst.plane_cfg.stride),
                    )
                },
                None => call_native(),
            },
            PixelType::U16 => call_native(),
        }
    }

    fn to_index(bsize: BlockSize) -> usize {
        bsize as usize & (DIST_FNS_LENGTH - 1)
    }

    fn get_satd_fns() -> &'static [Option<SatdFn>; DIST_FNS_LENGTH] {
        if is_x86_feature_detected!("avx2") {
            &SATD_FNS_AVX2
        } else if is_x86_feature_detected!("sse4.1") {
            &SATD_FNS_SSE4
        } else if is_x86_feature_detected!("ssse3") {
            &SATD_FNS_SSSE3
        } else {
            &[None; DIST_FNS_LENGTH]
        }
    }

    type SatdFn = unsafe extern "C" fn(
        src: *const u8,
        src_stride: isize,
        dst: *const u8,
        dst_stride: isize,
    ) -> u32;

    const DIST_FNS_LENGTH: usize = 32;

    static SATD_FNS_SSSE3: [Option<SatdFn>; DIST_FNS_LENGTH] = {
        let mut out: [Option<SatdFn>; DIST_FNS_LENGTH] = [None; DIST_FNS_LENGTH];
        use BlockSize::*;
        out[BLOCK_8X8 as usize] = Some(scenechangeasm_satd_8x8_ssse3);
        out
    };

    static SATD_FNS_SSE4: [Option<SatdFn>; DIST_FNS_LENGTH] = {
        let mut out: [Option<SatdFn>; DIST_FNS_LENGTH] = [None; DIST_FNS_LENGTH];
        use BlockSize::*;
        out[BLOCK_4X4 as usize] = Some(scenechangeasm_satd_4x4_sse4);
        out[BLOCK_8X8 as usize] = Some(scenechangeasm_satd_8x8_ssse3);
        out
    };

    static SATD_FNS_AVX2: [Option<SatdFn>; DIST_FNS_LENGTH] = {
        let mut out: [Option<SatdFn>; DIST_FNS_LENGTH] = [None; DIST_FNS_LENGTH];
        use BlockSize::*;
        out[BLOCK_4X4 as usize] = Some(scenechangeasm_satd_4x4_avx2);
        out[BLOCK_8X8 as usize] = Some(scenechangeasm_satd_8x8_avx2);
        out[BLOCK_16X16 as usize] = Some(scenechangeasm_satd_16x16_avx2);
        out[BLOCK_32X32 as usize] = Some(scenechangeasm_satd_32x32_avx2);
        out[BLOCK_64X64 as usize] = Some(scenechangeasm_satd_64x64_avx2);
        out[BLOCK_128X128 as usize] = Some(scenechangeasm_satd_128x128_avx2);
        out[BLOCK_4X8 as usize] = Some(scenechangeasm_satd_4x8_avx2);
        out[BLOCK_8X4 as usize] = Some(scenechangeasm_satd_8x4_avx2);
        out[BLOCK_8X16 as usize] = Some(scenechangeasm_satd_8x16_avx2);
        out[BLOCK_16X8 as usize] = Some(scenechangeasm_satd_16x8_avx2);
        out[BLOCK_16X32 as usize] = Some(scenechangeasm_satd_16x32_avx2);
        out[BLOCK_32X16 as usize] = Some(scenechangeasm_satd_32x16_avx2);
        out[BLOCK_32X64 as usize] = Some(scenechangeasm_satd_32x64_avx2);
        out[BLOCK_64X32 as usize] = Some(scenechangeasm_satd_64x32_avx2);
        out[BLOCK_64X128 as usize] = Some(scenechangeasm_satd_64x128_avx2);
        out[BLOCK_128X64 as usize] = Some(scenechangeasm_satd_128x64_avx2);
        out[BLOCK_4X16 as usize] = Some(scenechangeasm_satd_4x16_avx2);
        out[BLOCK_16X4 as usize] = Some(scenechangeasm_satd_16x4_avx2);
        out[BLOCK_8X32 as usize] = Some(scenechangeasm_satd_8x32_avx2);
        out[BLOCK_32X8 as usize] = Some(scenechangeasm_satd_32x8_avx2);
        out[BLOCK_16X64 as usize] = Some(scenechangeasm_satd_16x64_avx2);
        out[BLOCK_64X16 as usize] = Some(scenechangeasm_satd_64x16_avx2);
        out
    };
}
