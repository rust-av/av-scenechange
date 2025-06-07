use v_frame::pixel::Pixel;

use super::IntraEdge;
use crate::data::{block::TxSize, plane::PlaneRegionMut, prediction::PredictionVariant};

#[cfg_attr(
    all(asm_x86_64, any(target_feature = "ssse3", target_feature = "avx2")),
    cold
)]
pub(super) fn predict_dc_intra_internal<T: Pixel>(
    variant: PredictionVariant,
    dst: &mut PlaneRegionMut<'_, T>,
    tx_size: TxSize,
    bit_depth: usize,
    edge_buf: &IntraEdge<T>,
) {
    let width = tx_size.width();
    let height = tx_size.height();

    // left pixels are ordered from bottom to top and right-aligned
    let (left, _top_left, above) = edge_buf.as_slices();

    let above_slice = above;
    let left_slice = &left[left.len().saturating_sub(height)..];

    (match variant {
        PredictionVariant::NONE => pred_dc_128,
        PredictionVariant::LEFT => pred_dc_left,
        PredictionVariant::TOP => pred_dc_top,
        PredictionVariant::BOTH => pred_dc,
    })(dst, above_slice, left_slice, width, height, bit_depth)
}

fn pred_dc<T: Pixel>(
    output: &mut PlaneRegionMut<'_, T>,
    above: &[T],
    left: &[T],
    width: usize,
    height: usize,
    _bit_depth: usize,
) {
    let edges = left[..height].iter().chain(above[..width].iter());
    let len = (width + height) as u32;
    let avg = (edges.fold(0u32, |acc, &v| {
        let v: u32 = v.into();
        v + acc
    }) + (len >> 1))
        / len;
    let avg = T::cast_from(avg);

    for line in output.rows_iter_mut().take(height) {
        line[..width].fill(avg);
    }
}

fn pred_dc_128<T: Pixel>(
    output: &mut PlaneRegionMut<'_, T>,
    _above: &[T],
    _left: &[T],
    width: usize,
    height: usize,
    bit_depth: usize,
) {
    let v = T::cast_from(128u32 << (bit_depth - 8));
    for line in output.rows_iter_mut().take(height) {
        line[..width].fill(v);
    }
}

fn pred_dc_left<T: Pixel>(
    output: &mut PlaneRegionMut<'_, T>,
    _above: &[T],
    left: &[T],
    width: usize,
    height: usize,
    _bit_depth: usize,
) {
    let sum = left[..].iter().fold(0u32, |acc, &v| {
        let v: u32 = v.into();
        v + acc
    });
    let avg = T::cast_from((sum + (height >> 1) as u32) / height as u32);
    for line in output.rows_iter_mut().take(height) {
        line[..width].fill(avg);
    }
}

fn pred_dc_top<T: Pixel>(
    output: &mut PlaneRegionMut<'_, T>,
    above: &[T],
    _left: &[T],
    width: usize,
    height: usize,
    _bit_depth: usize,
) {
    let sum = above[..width].iter().fold(0u32, |acc, &v| {
        let v: u32 = v.into();
        v + acc
    });
    let avg = T::cast_from((sum + (width >> 1) as u32) / width as u32);
    for line in output.rows_iter_mut().take(height) {
        line[..width].fill(avg);
    }
}
