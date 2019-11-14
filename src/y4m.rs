use crate::pixel::*;
use crate::PlaneData;
use std::io::Read;
use std::mem;

fn copy_from_raw_u8<T: Pixel>(source: &[u8]) -> Vec<T> {
    match mem::size_of::<T>() {
        1 => source.iter().map(|byte| T::cast_from(*byte)).collect(),
        2 => {
            let mut output = Vec::with_capacity(source.len() / 2);
            for bytes in source.chunks(2) {
                output.push(T::cast_from(
                    u16::cast_from(bytes[1]) << 8 | u16::cast_from(bytes[0]),
                ));
            }
            output
        }
        _ => unreachable!(),
    }
}

pub(crate) fn read_video_frame<R: Read, T: Pixel>(
    dec: &mut y4m::Decoder<R>,
) -> Result<PlaneData<T>, ()> {
    dec.read_frame()
        .map(|frame| {
            [
                copy_from_raw_u8(frame.get_y_plane()),
                copy_from_raw_u8(frame.get_u_plane()),
                copy_from_raw_u8(frame.get_v_plane()),
            ]
        })
        .map_err(|_| ())
}
