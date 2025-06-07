use v_frame::pixel::{Pixel, PixelType};

use super::IntraEdge;
use crate::data::{block::TxSize, plane::PlaneRegionMut, prediction::PredictionVariant};

macro_rules! decl_angular_ipred_fn {
        ($($f:ident),+) => {
            extern "C" {
            $(
                fn $f(
                dst: *mut u8, stride: libc::ptrdiff_t, topleft: *const u8,
                width: libc::c_int, height: libc::c_int, angle: libc::c_int,
                );
            )*
            }
        };
    }

decl_angular_ipred_fn! {
    avsc_ipred_dc_8bpc_avx2,
    avsc_ipred_dc_left_8bpc_avx2,
    avsc_ipred_dc_128_8bpc_avx2,
    avsc_ipred_dc_top_8bpc_avx2
}

macro_rules! decl_angular_ipred_hbd_fn {
        ($($f:ident),+) => {
            extern "C" {
            $(
                fn $f(
                dst: *mut u16, stride: libc::ptrdiff_t, topleft: *const u16,
                width: libc::c_int, height: libc::c_int, angle: libc::c_int,
                max_width: libc::c_int, max_height: libc::c_int,
                bit_depth_max: libc::c_int,
                );
            )*
            }
        };
    }

decl_angular_ipred_hbd_fn! {
    avsc_ipred_dc_16bpc_avx2,
    avsc_ipred_dc_left_16bpc_avx2,
    avsc_ipred_dc_128_16bpc_avx2,
    avsc_ipred_dc_top_16bpc_avx2
}

#[target_feature(enable = "avx2")]
pub(super) fn predict_dc_intra_internal<T: Pixel>(
    variant: PredictionVariant,
    dst: &mut PlaneRegionMut<'_, T>,
    tx_size: TxSize,
    bit_depth: usize,
    edge_buf: &IntraEdge<T>,
) {
    // SAFETY: Calls Assembly code.
    unsafe {
        let stride = T::to_asm_stride(dst.plane_cfg.stride) as libc::ptrdiff_t;
        let w = tx_size.width() as libc::c_int;
        let h = tx_size.height() as libc::c_int;

        match T::type_enum() {
            PixelType::U8 => {
                let dst_ptr = dst.data_ptr_mut() as *mut _;
                let edge_ptr = edge_buf.top_left_ptr() as *const _;
                (match variant {
                    PredictionVariant::NONE => avsc_ipred_dc_128_8bpc_avx2,
                    PredictionVariant::LEFT => avsc_ipred_dc_left_8bpc_avx2,
                    PredictionVariant::TOP => avsc_ipred_dc_top_8bpc_avx2,
                    PredictionVariant::BOTH => avsc_ipred_dc_8bpc_avx2,
                })(dst_ptr, stride, edge_ptr, w, h, 0);
            }
            PixelType::U16 => {
                let dst_ptr = dst.data_ptr_mut() as *mut _;
                let edge_ptr = edge_buf.top_left_ptr() as *const _;
                let bd_max = (1 << bit_depth) - 1;
                (match variant {
                    PredictionVariant::NONE => avsc_ipred_dc_128_16bpc_avx2,
                    PredictionVariant::LEFT => avsc_ipred_dc_left_16bpc_avx2,
                    PredictionVariant::TOP => avsc_ipred_dc_top_16bpc_avx2,
                    PredictionVariant::BOTH => avsc_ipred_dc_16bpc_avx2,
                })(dst_ptr, stride, edge_ptr, w, h, 0, 0, 0, bd_max);
            }
        }
    }
}
