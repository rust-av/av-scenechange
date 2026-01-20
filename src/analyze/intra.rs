#[cfg(asm_x86_64)]
mod avx2;
#[cfg(asm_x86_64)]
mod avx512icl;
mod rust;
#[cfg(asm_x86_64)]
mod ssse3;

#[cfg(test)]
mod tests;

use std::{mem::MaybeUninit, num::NonZeroUsize};

use aligned::{A64, Aligned};
use cfg_if::cfg_if;
use v_frame::{frame::Frame, pixel::Pixel, plane::Plane};

use super::importance::IMPORTANCE_BLOCK_SIZE;
use crate::data::{
    block::{BlockSize, MAX_TX_SIZE, TxSize},
    plane::{Area, AsRegion, PlaneOffset, PlaneRegion, PlaneRegionMut, Rect},
    prediction::PredictionVariant,
    satd::get_satd,
    slice_assume_init_mut,
    superblock::MI_SIZE_LOG2,
    tile::TileRect,
};

pub const BLOCK_TO_PLANE_SHIFT: usize = MI_SIZE_LOG2;

pub(crate) fn estimate_intra_costs<T: Pixel>(
    temp_plane: &mut Plane<T>,
    frame: &Frame<T>,
    bit_depth: NonZeroUsize,
) -> Box<[u32]> {
    let plane = &frame.y_plane;
    let plane_after_prediction = temp_plane;

    let bsize = BlockSize::from_width_and_height(IMPORTANCE_BLOCK_SIZE, IMPORTANCE_BLOCK_SIZE);
    let tx_size = bsize.tx_size();

    let h_in_imp_b = plane.height().get() / IMPORTANCE_BLOCK_SIZE;
    let w_in_imp_b = plane.width().get() / IMPORTANCE_BLOCK_SIZE;
    let mut intra_costs = Vec::with_capacity(h_in_imp_b * w_in_imp_b);

    for y in 0..h_in_imp_b {
        for x in 0..w_in_imp_b {
            let plane_org = plane.region(Area::Rect(Rect {
                x: (x * IMPORTANCE_BLOCK_SIZE) as isize,
                y: (y * IMPORTANCE_BLOCK_SIZE) as isize,
                width: NonZeroUsize::new(IMPORTANCE_BLOCK_SIZE).expect("non-zero const"),
                height: NonZeroUsize::new(IMPORTANCE_BLOCK_SIZE).expect("non-zero const"),
            }));

            // For scene detection, we are only going to support DC_PRED
            // for simplicity and speed purposes.
            let mut edge_buf = Aligned([MaybeUninit::uninit(); 4 * MAX_TX_SIZE + 1]);
            let edge_buf = get_intra_edges(
                &mut edge_buf,
                &plane.as_region(),
                PlaneOffset {
                    x: (x * IMPORTANCE_BLOCK_SIZE) as isize,
                    y: (y * IMPORTANCE_BLOCK_SIZE) as isize,
                },
                bit_depth,
            );

            let mut plane_after_prediction_region =
                plane_after_prediction.region_mut(Area::Rect(Rect {
                    x: (x * IMPORTANCE_BLOCK_SIZE) as isize,
                    y: (y * IMPORTANCE_BLOCK_SIZE) as isize,
                    width: NonZeroUsize::new(IMPORTANCE_BLOCK_SIZE).expect("non-zero const"),
                    height: NonZeroUsize::new(IMPORTANCE_BLOCK_SIZE).expect("non-zero const"),
                }));

            predict_dc_intra(
                TileRect {
                    x: x * IMPORTANCE_BLOCK_SIZE,
                    y: y * IMPORTANCE_BLOCK_SIZE,
                    width: NonZeroUsize::new(IMPORTANCE_BLOCK_SIZE).expect("non-zero const"),
                    height: NonZeroUsize::new(IMPORTANCE_BLOCK_SIZE).expect("non-zero const"),
                },
                &mut plane_after_prediction_region,
                tx_size,
                bit_depth,
                &edge_buf,
            );

            let plane_after_prediction_region = plane_after_prediction.region(Area::Rect(Rect {
                x: (x * IMPORTANCE_BLOCK_SIZE) as isize,
                y: (y * IMPORTANCE_BLOCK_SIZE) as isize,
                width: NonZeroUsize::new(IMPORTANCE_BLOCK_SIZE).expect("non-zero const"),
                height: NonZeroUsize::new(IMPORTANCE_BLOCK_SIZE).expect("non-zero const"),
            }));

            let intra_cost = get_satd(
                &plane_org,
                &plane_after_prediction_region,
                bsize.width(),
                bsize.height(),
                bit_depth,
            );

            intra_costs.push(intra_cost);
        }
    }

    intra_costs.into_boxed_slice()
}

pub fn get_intra_edges<'a, T: Pixel>(
    edge_buf: &'a mut IntraEdgeBuffer<T>,
    dst: &PlaneRegion<'_, T>,
    po: PlaneOffset,
    bit_depth: NonZeroUsize,
) -> IntraEdge<'a, T> {
    let tx_size = TxSize::TX_8X8;
    let mut init_left: usize = 0;
    let mut init_above: usize = 0;

    let base = 128u16 << (bit_depth.get() - 8);

    {
        // left pixels are ordered from bottom to top and right-aligned
        let (left, not_left) = edge_buf.split_at_mut(2 * MAX_TX_SIZE);
        let (top_left, above) = not_left.split_at_mut(1);

        let x = po.x as usize;
        let y = po.y as usize;

        let needs_left = x != 0;
        let needs_top = y != 0;

        let rect_w = dst
            .rect()
            .width
            .get()
            .min(dst.plane_cfg.width.get() - dst.rect().x as usize);
        let rect_h = dst
            .rect()
            .height
            .get()
            .min(dst.plane_cfg.height.get() - dst.rect().y as usize);

        // Needs left
        if needs_left {
            let txh = if y + tx_size.height().get() > rect_h {
                rect_h - y
            } else {
                tx_size.height().get()
            };
            if x != 0 {
                for i in 0..txh {
                    debug_assert!(y + i < rect_h);
                    left[2 * MAX_TX_SIZE - 1 - i].write(dst[y + i][x - 1]);
                }
                if txh < tx_size.height().get() {
                    let val = dst[y + txh - 1][x - 1];
                    for i in txh..tx_size.height().get() {
                        left[2 * MAX_TX_SIZE - 1 - i].write(val);
                    }
                }
            } else {
                let val = if y != 0 {
                    dst[y - 1][0]
                } else {
                    T::from(base + 1).expect("value should fit in Pixel")
                };
                for v in left[2 * MAX_TX_SIZE - tx_size.height().get()..].iter_mut() {
                    v.write(val);
                }
            }
            init_left += tx_size.height().get();
        }

        // Needs top
        if needs_top {
            let txw = if x + tx_size.width().get() > rect_w {
                rect_w - x
            } else {
                tx_size.width().get()
            };
            if y != 0 {
                above[..txw].copy_from_slice(
                    // SAFETY: &[T] and &[MaybeUninit<T>] have the same layout
                    unsafe {
                        &*(&dst[y - 1][x..x + txw] as *const [T]
                            as *const [std::mem::MaybeUninit<T>])
                    },
                );
                if txw < tx_size.width().get() {
                    let val = dst[y - 1][x + txw - 1];
                    for v in &mut above[txw..tx_size.width().get()] {
                        v.write(val);
                    }
                }
            } else {
                let val = if x != 0 {
                    dst[0][x - 1]
                } else {
                    T::from(base - 1).expect("value should fit in Pixel")
                };
                for v in &mut above[..tx_size.width().get()] {
                    v.write(val);
                }
            }
            init_above += tx_size.width().get();
        }

        top_left[0].write(T::from(base).expect("value should fit in Pixel"));
    }
    IntraEdge::new(edge_buf, init_left, init_above)
}

pub fn predict_dc_intra<T: Pixel>(
    tile_rect: TileRect,
    dst: &mut PlaneRegionMut<'_, T>,
    tx_size: TxSize,
    bit_depth: NonZeroUsize,
    edge_buf: &IntraEdge<T>,
) {
    let &Rect {
        x: frame_x,
        y: frame_y,
        ..
    } = dst.rect();
    debug_assert!(frame_x >= 0 && frame_y >= 0);
    // x and y are expressed relative to the tile
    let x = frame_x as usize - tile_rect.x;
    let y = frame_y as usize - tile_rect.y;

    let variant = PredictionVariant::new(x, y);

    cfg_if! {
        if #[cfg(asm_x86_64)] {
            // There is currently a crash in the HBD ASM when the `dst` width is not mod 8.
            // Fallback to Rust code for that case.
            if !(size_of::<T>() == 2 && dst.plane_cfg.width.get() % 8 > 0) {
                if crate::cpu::has_avx512icl() {
                    // SAFETY: call to SIMD function
                    unsafe { avx512icl::predict_dc_intra_internal(variant, dst, tx_size, bit_depth, edge_buf); }
                    return;
                } else if crate::cpu::has_avx2() {
                    // SAFETY: call to SIMD function
                    unsafe { avx2::predict_dc_intra_internal(variant, dst, tx_size, bit_depth, edge_buf); }
                    return;
                } else if crate::cpu::has_ssse3() {
                    // SAFETY: call to SIMD function
                    unsafe { ssse3::predict_dc_intra_internal(variant, dst, tx_size, bit_depth, edge_buf); }
                    return;
                }
            }
        }
    }

    rust::predict_dc_intra_internal::<T>(variant, dst, tx_size, bit_depth, edge_buf);
}

type IntraEdgeBuffer<T> = Aligned<A64, [MaybeUninit<T>; 4 * MAX_TX_SIZE + 1]>;

pub struct IntraEdge<'a, T: Pixel>(&'a [T], &'a [T], &'a [T]);

impl<'a, T: Pixel> IntraEdge<'a, T> {
    fn new(edge_buf: &'a mut IntraEdgeBuffer<T>, init_left: usize, init_above: usize) -> Self {
        // SAFETY: Initialized in `get_intra_edges`.
        let left = unsafe {
            let begin_left = 2 * MAX_TX_SIZE - init_left;
            let end_above = 2 * MAX_TX_SIZE + 1 + init_above;
            slice_assume_init_mut(&mut edge_buf[begin_left..end_above])
        };
        let (left, top_left) = left.split_at(init_left);
        let (top_left, above) = top_left.split_at(1);
        Self(left, top_left, above)
    }

    pub const fn as_slices(&self) -> (&'a [T], &'a [T], &'a [T]) {
        (self.0, self.1, self.2)
    }

    #[allow(dead_code)]
    pub const fn top_left_ptr(&self) -> *const T {
        self.1.as_ptr()
    }
}
