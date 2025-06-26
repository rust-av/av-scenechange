use std::{collections::HashMap, mem::size_of, path::Path, slice};

use anyhow::{bail, ensure};
use num_rational::Rational32;
use v_frame::{
    frame::Frame,
    pixel::{ChromaSampling, Pixel},
};
use vapoursynth::{
    api::API,
    core::CoreRef,
    map::OwnedMap,
    node::Node,
    video_info::{Property, VideoInfo},
    vsscript::{Environment, EvalFlags},
};

use crate::decoder::VideoDetails;

const OUTPUT_INDEX: i32 = 0;

pub struct VapoursynthDecoder<'core> {
    // env: &'core Environment,
    pub core: CoreRef<'core>,
    pub node: Node<'core>,
    frames_read: usize,
    total_frames: usize,
}

impl<'core> VapoursynthDecoder<'core> {
    /// # Errors
    ///
    /// - If sourcing an invalid Vapoursynth script.
    /// - If using a Vapoursynth script that contains an unsupported video
    ///   format.
    #[inline]
    pub fn new(env: &'core Environment) -> anyhow::Result<Self> {
        let core = env.get_core()?;
        let node = env.get_output(OUTPUT_INDEX)?.0;

        let total_frames = get_num_frames(node.info())?;

        Ok(Self {
            // env,
            core,
            node,
            frames_read: 0,
            total_frames,
        })
    }

    /// # Errors
    ///
    /// - If sourcing an invalid Vapoursynth script.
    /// - If using a Vapoursynth script that contains an unsupported video
    ///   format.
    /// - If arguments are invalid.
    #[inline]
    pub fn new_from_file(
        env: &'core mut Environment,
        source: &Path,
        arguments: Option<HashMap<String, String>>,
    ) -> anyhow::Result<Self> {
        // Set arguments
        VapoursynthDecoder::set_arguments(env, arguments)?;
        env.eval_file(source, EvalFlags::SetWorkingDir)?;

        Self::new(env)
    }

    /// # Errors
    ///
    /// - If Vapoursynth script is invalid.
    /// - If using a Vapoursynth script that contains an unsupported video
    ///   format.
    /// - If arguments are invalid.
    #[inline]
    pub fn new_from_script(
        env: &'core mut Environment,
        script: &str,
        arguments: Option<HashMap<String, String>>,
    ) -> anyhow::Result<Self> {
        VapoursynthDecoder::set_arguments(env, arguments)?;

        env.eval_script(script)?;

        Self::new(env)
    }

    fn set_arguments(
        env: &Environment,
        arguments: Option<HashMap<String, String>>,
    ) -> anyhow::Result<()> {
        let args_err_msg = "Failed to set arguments";
        let api = API::get().ok_or(anyhow::anyhow!(args_err_msg))?;
        let mut arguments_map = OwnedMap::new(api);

        if let Some(arguments) = arguments {
            for (key, value) in arguments {
                arguments_map
                    .set_data(key.as_str(), value.as_bytes())
                    .map_err(|_| anyhow::anyhow!(args_err_msg))?;
            }
        }

        Ok(env.set_variables(&arguments_map)?)
    }

    /// # Errors
    ///
    /// - If sourcing an invalid Vapoursynth script.
    /// - If using a Vapoursynth script that contains an unsupported video
    ///   format.
    #[inline]
    pub fn get_video_details(&self) -> anyhow::Result<VideoDetails> {
        // let (node, _) = self.env.get_output(OUTPUT_INDEX)?;
        let info = self.node.info();
        let (width, height) = get_resolution(info)?;
        Ok(VideoDetails {
            width,
            height,
            bit_depth: get_bit_depth(info)?,
            chroma_sampling: get_chroma_sampling(info)?,
            time_base: get_time_base(info)?,
        })
    }

    /// # Errors
    ///
    /// - If sourcing an invalid Vapoursynth script.
    /// - If using a Vapoursynth script that contains an unsupported video
    ///   format.
    /// - If a frame cannot be read.
    #[allow(clippy::transmute_ptr_to_ptr)]
    #[inline]
    pub fn read_video_frame<T: Pixel>(&mut self, cfg: &VideoDetails) -> anyhow::Result<Frame<T>> {
        const SB_SIZE_LOG2: usize = 6;
        const SB_SIZE: usize = 1 << SB_SIZE_LOG2;
        const SUBPEL_FILTER_SIZE: usize = 8;
        const FRAME_MARGIN: usize = 16 + SUBPEL_FILTER_SIZE;
        const LUMA_PADDING: usize = SB_SIZE + FRAME_MARGIN;

        if self.frames_read >= self.total_frames {
            bail!("No frames left");
        }

        let vs_frame = &self.node.get_frame(self.frames_read)?;
        self.frames_read += 1;

        let bytes = size_of::<T>();
        let mut f: Frame<T> =
            Frame::new_with_padding(cfg.width, cfg.height, cfg.chroma_sampling, LUMA_PADDING);

        // SAFETY: We are using the stride to compute the length of the data slice
        unsafe {
            f.planes[0].copy_from_raw_u8(
                slice::from_raw_parts(
                    vs_frame.data_ptr(0),
                    vs_frame.stride(0) * vs_frame.height(0),
                ),
                vs_frame.stride(0),
                bytes,
            );
            f.planes[1].copy_from_raw_u8(
                slice::from_raw_parts(
                    vs_frame.data_ptr(1),
                    vs_frame.stride(1) * vs_frame.height(1),
                ),
                vs_frame.stride(1),
                bytes,
            );
            f.planes[2].copy_from_raw_u8(
                slice::from_raw_parts(
                    vs_frame.data_ptr(2),
                    vs_frame.stride(2) * vs_frame.height(2),
                ),
                vs_frame.stride(2),
                bytes,
            );
        }
        Ok(f)
    }
}

/// Get the number of frames from a Vapoursynth `VideoInfo` struct.
fn get_num_frames(info: VideoInfo) -> anyhow::Result<usize> {
    let num_frames = {
        if Property::Variable == info.format {
            bail!("Cannot output clips with varying format");
        }
        if Property::Variable == info.resolution {
            bail!("Cannot output clips with varying dimensions");
        }
        if Property::Variable == info.framerate {
            bail!("Cannot output clips with varying framerate");
        }

        info.num_frames
    };

    ensure!(num_frames != 0, "vapoursynth reported 0 frames");

    Ok(num_frames)
}

/// Get the bit depth from a Vapoursynth `VideoInfo` struct.
fn get_bit_depth(info: VideoInfo) -> anyhow::Result<usize> {
    let bits_per_sample = {
        match info.format {
            Property::Variable => {
                bail!("Cannot output clips with variable format");
            }
            Property::Constant(x) => x.bits_per_sample(),
        }
    };

    Ok(bits_per_sample as usize)
}

/// Get the resolution from a Vapoursynth `VideoInfo` struct.
fn get_resolution(info: VideoInfo) -> anyhow::Result<(usize, usize)> {
    let resolution = {
        match info.resolution {
            Property::Variable => {
                bail!("Cannot output clips with variable resolution");
            }
            Property::Constant(x) => x,
        }
    };

    Ok((resolution.width, resolution.height))
}

/// Get the time base (inverse of frame rate) from a Vapoursynth `VideoInfo`
/// struct.
fn get_time_base(info: VideoInfo) -> anyhow::Result<Rational32> {
    match info.framerate {
        Property::Variable => bail!("Cannot output clips with varying framerate"),
        Property::Constant(fps) => Ok(Rational32::new(
            fps.denominator as i32,
            fps.numerator as i32,
        )),
    }
}

/// Get the chroma sampling from a Vapoursynth `VideoInfo` struct.
fn get_chroma_sampling(info: VideoInfo) -> anyhow::Result<ChromaSampling> {
    match info.format {
        Property::Variable => bail!("Variable pixel format not supported"),
        Property::Constant(x) => match x.color_family() {
            vapoursynth::format::ColorFamily::YUV => {
                let ss = (x.sub_sampling_w(), x.sub_sampling_h());
                match ss {
                    (1, 1) => Ok(ChromaSampling::Cs420),
                    (1, 0) => Ok(ChromaSampling::Cs422),
                    (0, 0) => Ok(ChromaSampling::Cs444),
                    _ => bail!("Unrecognized chroma subsampling"),
                }
            }
            vapoursynth::format::ColorFamily::Gray => Ok(ChromaSampling::Cs400),
            _ => bail!("Currently only YUV input is supported"),
        },
    }
}
