use std::io::Read;

use rav1e::prelude::{ChromaSamplePosition, ChromaSampling, Frame, Pixel, Rational};

#[cfg(feature = "ffmpeg")]
use crate::ffmpeg::FfmpegDecoder;
#[cfg(feature = "vapoursynth")]
use crate::vapoursynth::VapoursynthDecoder;

pub enum Decoder<R: Read> {
    Y4m(y4m::Decoder<R>),
    #[cfg(feature = "vapoursynth")]
    Vapoursynth(VapoursynthDecoder),
    #[cfg(feature = "ffmpeg")]
    Ffmpeg(FfmpegDecoder),
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
            #[cfg(feature = "ffmpeg")]
            Decoder::Ffmpeg(dec) => Ok(dec.video_details),
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
            #[cfg(feature = "ffmpeg")]
            Decoder::Ffmpeg(dec) => dec.read_video_frame::<T>(),
        }
    }
}

#[derive(Debug, Clone, Copy)]
pub struct VideoDetails {
    pub width: usize,
    pub height: usize,
    pub bit_depth: usize,
    pub chroma_sampling: ChromaSampling,
    pub chroma_sample_position: ChromaSamplePosition,
    pub time_base: Rational,
}

impl Default for VideoDetails {
    fn default() -> Self {
        VideoDetails {
            width: 640,
            height: 480,
            bit_depth: 8,
            chroma_sampling: ChromaSampling::Cs420,
            chroma_sample_position: ChromaSamplePosition::Unknown,
            time_base: Rational { num: 30, den: 1 },
        }
    }
}
