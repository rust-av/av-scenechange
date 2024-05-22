use std::io::Read;

use rav1e::{Frame, Pixel};

#[cfg(feature = "vapoursynth")]
use crate::vapoursynth::VapoursynthDecoder;
use crate::y4m::VideoDetails;

pub enum Decoder<R: Read> {
    Y4m(y4m::Decoder<R>),
    #[cfg(feature = "vapoursynth")]
    Vapoursynth(VapoursynthDecoder),
}

impl<R: Read> Decoder<R> {
    /// # Errors
    ///
    /// - If using a Vapoursynth script that contains an unsupported video format.
    pub fn get_video_details(&self) -> anyhow::Result<VideoDetails> {
        match self {
            Decoder::Y4m(dec) => Ok(crate::y4m::get_video_details(dec)),
            #[cfg(feature = "vapoursynth")]
            Decoder::Vapoursynth(dec) => dec.get_video_details(),
        }
    }

    /// # Errors
    ///
    /// - If a frame cannot be read.
    pub fn read_video_frame<T: Pixel>(
        &mut self,
        video_details: &VideoDetails,
    ) -> anyhow::Result<Frame<T>> {
        match self {
            Decoder::Y4m(dec) => crate::y4m::read_video_frame::<R, T>(dec, video_details),
            #[cfg(feature = "vapoursynth")]
            Decoder::Vapoursynth(dec) => dec.read_video_frame::<T>(video_details),
        }
    }
}
