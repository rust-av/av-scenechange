use std::num::NonZeroUsize;

use v_frame::pixel::Pixel;

use super::IntraEdge;
use crate::data::{block::TxSize, plane::PlaneRegionMut, prediction::PredictionVariant};

pub(super) fn predict_dc_intra_internal<T: Pixel>(
    variant: PredictionVariant,
    dst: &mut PlaneRegionMut<'_, T>,
    tx_size: TxSize,
    bit_depth: NonZeroUsize,
    edge_buf: &IntraEdge<T>,
) {
    let width = tx_size.width();
    let height = tx_size.height();

    // left pixels are ordered from bottom to top and right-aligned
    let (left, _top_left, above) = edge_buf.as_slices();

    let above_slice = above;
    let left_slice = &left[left.len().saturating_sub(height.get())..];

    (match variant {
        PredictionVariant::NONE => pred_dc_128,
        PredictionVariant::LEFT => pred_dc_left,
        PredictionVariant::TOP => pred_dc_top,
        PredictionVariant::BOTH => pred_dc,
    })(dst, above_slice, left_slice, width, height, bit_depth);
}

fn pred_dc<T: Pixel>(
    output: &mut PlaneRegionMut<'_, T>,
    above: &[T],
    left: &[T],
    width: NonZeroUsize,
    height: NonZeroUsize,
    _bit_depth: NonZeroUsize,
) {
    let edges = left[..(height.get())]
        .iter()
        .chain(above[..(width.get())].iter());
    let len = (width.get() + height.get()) as u32;
    let avg = (edges.fold(0u32, |acc, &v| {
        let v: u32 = v.to_u32().expect("value should fit in u32");
        v + acc
    }) + (len >> 1))
        / len;
    let avg = T::from(avg).expect("value should fit in Pixel");

    assert!(output.rect.width >= width);
    for line in output.rows_iter_mut().take(height.get()) {
        // SAFETY: bounds are asserted above
        unsafe {
            line.get_unchecked_mut(..width.get()).fill(avg);
        }
    }
}

fn pred_dc_128<T: Pixel>(
    output: &mut PlaneRegionMut<'_, T>,
    _above: &[T],
    _left: &[T],
    width: NonZeroUsize,
    height: NonZeroUsize,
    bit_depth: NonZeroUsize,
) {
    let v = T::from(128u32 << (bit_depth.get() - 8)).expect("value should fit in Pixel");

    assert!(output.rect.width >= width);
    for line in output.rows_iter_mut().take(height.get()) {
        // SAFETY: bounds are asserted above
        unsafe {
            line.get_unchecked_mut(..width.get()).fill(v);
        }
    }
}

fn pred_dc_left<T: Pixel>(
    output: &mut PlaneRegionMut<'_, T>,
    _above: &[T],
    left: &[T],
    width: NonZeroUsize,
    height: NonZeroUsize,
    _bit_depth: NonZeroUsize,
) {
    let sum = left.iter().fold(0u32, |acc, &v| {
        let v: u32 = v.to_u32().expect("value should fit in u32");
        v + acc
    });
    let avg = T::from((sum + (height.get() >> 1) as u32) / height.get() as u32)
        .expect("value should fit in Pixel");

    assert!(output.rect.width >= width);
    for line in output.rows_iter_mut().take(height.get()) {
        // SAFETY: bounds are asserted above
        unsafe {
            line.get_unchecked_mut(..width.get()).fill(avg);
        }
    }
}

fn pred_dc_top<T: Pixel>(
    output: &mut PlaneRegionMut<'_, T>,
    above: &[T],
    _left: &[T],
    width: NonZeroUsize,
    height: NonZeroUsize,
    _bit_depth: NonZeroUsize,
) {
    let sum = above[..(width.get())].iter().fold(0u32, |acc, &v| {
        let v: u32 = v.to_u32().expect("value should fit in u32");
        v + acc
    });
    let avg = T::from((sum + (width.get() >> 1) as u32) / width.get() as u32)
        .expect("value should fit in Pixel");

    assert!(output.rect.width >= width);
    for line in output.rows_iter_mut().take(height.get()) {
        // SAFETY: bounds are asserted above
        unsafe {
            line.get_unchecked_mut(..width.get()).fill(avg);
        }
    }
}
