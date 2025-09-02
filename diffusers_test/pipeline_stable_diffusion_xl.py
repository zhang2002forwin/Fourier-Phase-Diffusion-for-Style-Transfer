# Copyright 2023 The HuggingFace Team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import inspect
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import torch
from torch import optim
from transformers import CLIPTextModel, CLIPTextModelWithProjection, CLIPTokenizer, CLIPVisionModel,CLIPVisionConfig,CLIPVisionModelWithProjection,CLIPProcessor, CLIPModel, CLIPImageProcessor

from diffusers.image_processor import VaeImageProcessor
from diffusers.loaders import FromSingleFileMixin, LoraLoaderMixin, TextualInversionLoaderMixin
from diffusers.models import AutoencoderKL, UNet2DConditionModel
from diffusers.models.attention_processor import (
    AttnProcessor2_0,
    LoRAAttnProcessor2_0,
    LoRAXFormersAttnProcessor,
    XFormersAttnProcessor,
)
from diffusers.schedulers import KarrasDiffusionSchedulers
from diffusers.utils import (
    is_accelerate_available,
    is_accelerate_version,
    logging,
    randn_tensor,
    replace_example_docstring,
)
from diffusers.pipeline_utils import DiffusionPipeline
from diffusers.pipelines.stable_diffusion_xl import StableDiffusionXLPipelineOutput
import PIL.Image
from PIL import Image, ImageFilter
import numpy as np
import sys
from tqdm import tqdm
import math
import matplotlib.pyplot as plt
import seaborn as sns
import cv2

logger = logging.get_logger(__name__)  # pylint: disable=invalid-name

EXAMPLE_DOC_STRING = """
    Examples:
        ```py
        >>> import torch
        >>> from diffusers import StableDiffusionXLPipeline

        >>> pipe = StableDiffusionXLPipeline.from_pretrained(
        ...     "stabilityai/stable-diffusion-xl-base-0.9", torch_dtype=torch.float16
        ... )
        >>> pipe = pipe.to("cuda")

        >>> prompt = "a photo of an astronaut riding a horse on mars"
        >>> image = pipe(prompt).images[0]
        ```
"""


def rescale_noise_cfg(noise_cfg, noise_pred_text, guidance_rescale=0.0):
    """
    Rescale `noise_cfg` according to `guidance_rescale`. Based on findings of [Common Diffusion Noise Schedules and
    Sample Steps are Flawed](https://arxiv.org/pdf/2305.08891.pdf). See Section 3.4
    """
    std_text = noise_pred_text.std(dim=list(range(1, noise_pred_text.ndim)), keepdim=True)
    std_cfg = noise_cfg.std(dim=list(range(1, noise_cfg.ndim)), keepdim=True)
    # rescale the results from guidance (fixes overexposure)
    noise_pred_rescaled = noise_cfg * (std_text / std_cfg)
    # mix with the original results from guidance by factor guidance_rescale to avoid "plain looking" images
    noise_cfg = guidance_rescale * noise_pred_rescaled + (1 - guidance_rescale) * noise_cfg
    return noise_cfg


class StableDiffusionXLPipeline(DiffusionPipeline, FromSingleFileMixin):
    r"""
    Pipeline for text-to-image generation using Stable Diffusion.

    This model inherits from [`DiffusionPipeline`]. Check the superclass documentation for the generic methods the
    library implements for all the pipelines (such as downloading or saving, running on a particular device, etc.)

    In addition the pipeline inherits the following loading methods:
        - *Textual-Inversion*: [`loaders.TextualInversionLoaderMixin.load_textual_inversion`]
        - *LoRA*: [`loaders.LoraLoaderMixin.load_lora_weights`]
        - *Ckpt*: [`loaders.FromSingleFileMixin.from_single_file`]

    as well as the following saving methods:
        - *LoRA*: [`loaders.LoraLoaderMixin.save_lora_weights`]

    Args:
        vae ([`AutoencoderKL`]):
            Variational Auto-Encoder (VAE) Model to encode and decode images to and from latent representations.
        text_encoder ([`CLIPTextModel`]):
            Frozen text-encoder. Stable Diffusion uses the text portion of
            [CLIP](https://huggingface.co/docs/transformers/model_doc/clip#transformers.CLIPTextModel), specifically
            the [clip-vit-large-patch14](https://huggingface.co/openai/clip-vit-large-patch14) variant.
        tokenizer (`CLIPTokenizer`):
            Tokenizer of class
            [CLIPTokenizer](https://huggingface.co/docs/transformers/v4.21.0/en/model_doc/clip#transformers.CLIPTokenizer).
        unet ([`UNet2DConditionModel`]): Conditional U-Net architecture to denoise the encoded image latents.
        scheduler ([`SchedulerMixin`]):
            A scheduler to be used in combination with `unet` to denoise the encoded image latents. Can be one of
            [`DDIMScheduler`], [`LMSDiscreteScheduler`], or [`PNDMScheduler`].
    """

    def __init__(
        self,
        vae: AutoencoderKL,
        text_encoder: CLIPTextModel,
        text_encoder_2: CLIPTextModelWithProjection,
        tokenizer: CLIPTokenizer,
        tokenizer_2: CLIPTokenizer,
        unet: UNet2DConditionModel,
        scheduler: KarrasDiffusionSchedulers,
        force_zeros_for_empty_prompt: bool = True,
    ):
        super().__init__()

        self.register_modules(
            vae=vae,
            text_encoder=text_encoder,
            text_encoder_2=text_encoder_2,
            tokenizer=tokenizer,
            tokenizer_2=tokenizer_2,
            unet=unet,
            scheduler=scheduler,
        )
        configuration = CLIPVisionConfig()
        #self.img_encoder=CLIPVisionModel.from_pretrained("/mnt/CV_teamz/pretrained/clip-vit-large-patch14", local_files_only=True)
        #self.processor_l = CLIPProcessor.from_pretrained("/mnt/CV_teamz/pretrained/clip-vit-large-patch14")
        #self.model_l = CLIPModel.from_pretrained("/mnt/CV_teamz/pretrained/clip-vit-large-patch14")
        #self.img_encoder_2=CLIPVisionModelWithProjection.from_pretrained("/mnt/CV_teamz/pretrained/CLIP-ViT-bigG-14-laion2B-39B-b160k/text_encoder",local_files_only=True,ignore_mismatched_sizes=True)
        #self.processor_bigG = CLIPProcessor.from_pretrained("/mnt/CV_teamz/pretrained/CLIP-bigG")
        #self.model_bigG = CLIPModel.from_pretrained("/mnt/CV_teamz/pretrained/CLIP-bigG")
        
        #self.model1 = CLIPModel.from_pretrained('/mnt/CV_teamz/pretrained/clip-vit-large-patch14/')
        #self.model2 = CLIPModel.from_pretrained('/mnt/CV_teamz/pretrained/CLIP-ViT-bigG-14-laion2B-39B-b160k-full')
        #self.model1.requires_grad_(False)
        #self.model2.requires_grad_(False)
        
        self.register_to_config(force_zeros_for_empty_prompt=force_zeros_for_empty_prompt)
        self.vae_scale_factor = 2 ** (len(self.vae.config.block_out_channels) - 1)
        self.image_processor = VaeImageProcessor(vae_scale_factor=self.vae_scale_factor)
        self.default_sample_size = self.unet.config.sample_size
        self.refimg=None
        self.t = None
        self.replace_1st = True
        self.adain_with_patches = False
    # Copied from diffusers.pipelines.stable_diffusion.pipeline_stable_diffusion.StableDiffusionPipeline.enable_vae_slicing
    def enable_vae_slicing(self):
        r"""
        Enable sliced VAE decoding.

        When this option is enabled, the VAE will split the input tensor in slices to compute decoding in several
        steps. This is useful to save some memory and allow larger batch sizes.
        """
        self.vae.enable_slicing()

    # Copied from diffusers.pipelines.stable_diffusion.pipeline_stable_diffusion.StableDiffusionPipeline.disable_vae_slicing
    def disable_vae_slicing(self):
        r"""
        Disable sliced VAE decoding. If `enable_vae_slicing` was previously invoked, this method will go back to
        computing decoding in one step.
        """
        self.vae.disable_slicing()

    # Copied from diffusers.pipelines.stable_diffusion.pipeline_stable_diffusion.StableDiffusionPipeline.enable_vae_tiling
    def enable_vae_tiling(self):
        r"""
        Enable tiled VAE decoding.

        When this option is enabled, the VAE will split the input tensor into tiles to compute decoding and encoding in
        several steps. This is useful to save a large amount of memory and to allow the processing of larger images.
        """
        self.vae.enable_tiling()

    # Copied from diffusers.pipelines.stable_diffusion.pipeline_stable_diffusion.StableDiffusionPipeline.disable_vae_tiling
    def disable_vae_tiling(self):
        r"""
        Disable tiled VAE decoding. If `enable_vae_tiling` was previously invoked, this method will go back to
        computing decoding in one step.
        """
        self.vae.disable_tiling()

    def enable_sequential_cpu_offload(self, gpu_id=0):
        r"""
        Offloads all models to CPU using accelerate, significantly reducing memory usage. When called, unet,
        text_encoder, vae and safety checker have their state dicts saved to CPU and then are moved to a
        `torch.device('meta') and loaded to GPU only when their specific submodule has its `forward` method called.
        Note that offloading happens on a submodule basis. Memory savings are higher than with
        `enable_model_cpu_offload`, but performance is lower.
        """
        if is_accelerate_available() and is_accelerate_version(">=", "0.14.0"):
            from accelerate import cpu_offload
        else:
            raise ImportError("`enable_sequential_cpu_offload` requires `accelerate v0.14.0` or higher")

        device = torch.device(f"cuda:{gpu_id}")

        if self.device.type != "cpu":
            self.to("cpu", silence_dtype_warnings=True)
            torch.cuda.empty_cache()  # otherwise we don't see the memory savings (but they probably exist)

        for cpu_offloaded_model in [self.unet, self.text_encoder, self.text_encoder_2, self.vae]:
            cpu_offload(cpu_offloaded_model, device)

    def enable_model_cpu_offload(self, gpu_id=0):
        r"""
        Offloads all models to CPU using accelerate, reducing memory usage with a low impact on performance. Compared
        to `enable_sequential_cpu_offload`, this method moves one whole model at a time to the GPU when its `forward`
        method is called, and the model remains in GPU until the next model runs. Memory savings are lower than with
        `enable_sequential_cpu_offload`, but performance is much better due to the iterative execution of the `unet`.
        """
        if is_accelerate_available() and is_accelerate_version(">=", "0.17.0.dev0"):
            from accelerate import cpu_offload_with_hook
        else:
            raise ImportError("`enable_model_cpu_offload` requires `accelerate v0.17.0` or higher.")

        device = torch.device(f"cuda:{gpu_id}")

        if self.device.type != "cpu":
            self.to("cpu", silence_dtype_warnings=True)
            torch.cuda.empty_cache()  # otherwise we don't see the memory savings (but they probably exist)

        model_sequence = (
            [self.text_encoder, self.text_encoder_2] if self.text_encoder is not None else [self.text_encoder_2]
        )
        model_sequence.extend([self.unet, self.vae])

        hook = None
        for cpu_offloaded_model in model_sequence:
            _, hook = cpu_offload_with_hook(cpu_offloaded_model, device, prev_module_hook=hook)

        # We'll offload the last model manually.
        self.final_offload_hook = hook

    @property
    # Copied from diffusers.pipelines.stable_diffusion.pipeline_stable_diffusion.StableDiffusionPipeline._execution_device
    def _execution_device(self):
        r"""
        Returns the device on which the pipeline's models will be executed. After calling
        `pipeline.enable_sequential_cpu_offload()` the execution device can only be inferred from Accelerate's module
        hooks.
        """
        if not hasattr(self.unet, "_hf_hook"):
            return self.device
        for module in self.unet.modules():
            if (
                hasattr(module, "_hf_hook")
                and hasattr(module._hf_hook, "execution_device")
                and module._hf_hook.execution_device is not None
            ):
                return torch.device(module._hf_hook.execution_device)
        return self.device

    def encode_prompt(
        self,
        prompt,
        device: Optional[torch.device] = None,
        num_images_per_prompt: int = 1,
        do_classifier_free_guidance: bool = True,
        negative_prompt=None,
        prompt_embeds: Optional[torch.FloatTensor] = None,
        negative_prompt_embeds: Optional[torch.FloatTensor] = None,
        pooled_prompt_embeds: Optional[torch.FloatTensor] = None,
        negative_pooled_prompt_embeds: Optional[torch.FloatTensor] = None,
        lora_scale: Optional[float] = None,
    ):
        r"""
        Encodes the prompt into text encoder hidden states.

        Args:
             prompt (`str` or `List[str]`, *optional*):
                prompt to be encoded
            device: (`torch.device`):
                torch device
            num_images_per_prompt (`int`):
                number of images that should be generated per prompt
            do_classifier_free_guidance (`bool`):
                whether to use classifier free guidance or not
            negative_prompt (`str` or `List[str]`, *optional*):
                The prompt or prompts not to guide the image generation. If not defined, one has to pass
                `negative_prompt_embeds` instead. Ignored when not using guidance (i.e., ignored if `guidance_scale` is
                less than `1`).
            prompt_embeds (`torch.FloatTensor`, *optional*):
                Pre-generated text embeddings. Can be used to easily tweak text inputs, *e.g.* prompt weighting. If not
                provided, text embeddings will be generated from `prompt` input argument.
            negative_prompt_embeds (`torch.FloatTensor`, *optional*):
                Pre-generated negative text embeddings. Can be used to easily tweak text inputs, *e.g.* prompt
                weighting. If not provided, negative_prompt_embeds will be generated from `negative_prompt` input
                argument.
            pooled_prompt_embeds (`torch.FloatTensor`, *optional*):
                Pre-generated pooled text embeddings. Can be used to easily tweak text inputs, *e.g.* prompt weighting.
                If not provided, pooled text embeddings will be generated from `prompt` input argument.
            negative_pooled_prompt_embeds (`torch.FloatTensor`, *optional*):
                Pre-generated negative pooled text embeddings. Can be used to easily tweak text inputs, *e.g.* prompt
                weighting. If not provided, pooled negative_prompt_embeds will be generated from `negative_prompt`
                input argument.
            lora_scale (`float`, *optional*):
                A lora scale that will be applied to all LoRA layers of the text encoder if LoRA layers are loaded.
        """
        device = device or self._execution_device

        # set lora scale so that monkey patched LoRA
        # function of text encoder can correctly access it
        if lora_scale is not None and isinstance(self, LoraLoaderMixin):
            self._lora_scale = lora_scale

        if prompt is not None and isinstance(prompt, str):
            batch_size = 1
        elif prompt is not None and isinstance(prompt, list):
            batch_size = len(prompt)
        else:
            batch_size = prompt_embeds.shape[0]

        # Define tokenizers and text encoders
        tokenizers = [self.tokenizer, self.tokenizer_2] if self.tokenizer is not None else [self.tokenizer_2]
        text_encoders = (
            [self.text_encoder, self.text_encoder_2] if self.text_encoder is not None else [self.text_encoder_2]
        )

        if prompt_embeds is None:
            # textual inversion: procecss multi-vector tokens if necessary
            prompt_embeds_list = []
            for tokenizer, text_encoder in zip(tokenizers, text_encoders):
                if isinstance(self, TextualInversionLoaderMixin):
                    prompt = self.maybe_convert_prompt(prompt, tokenizer)

                text_inputs = tokenizer(
                    prompt,
                    padding="max_length",
                    max_length=tokenizer.model_max_length,
                    truncation=True,
                    return_tensors="pt",
                )
                text_input_ids = text_inputs.input_ids
                untruncated_ids = tokenizer(prompt, padding="longest", return_tensors="pt").input_ids

                if untruncated_ids.shape[-1] >= text_input_ids.shape[-1] and not torch.equal(
                    text_input_ids, untruncated_ids
                ):
                    removed_text = tokenizer.batch_decode(untruncated_ids[:, tokenizer.model_max_length - 1 : -1])
                    logger.warning(
                        "The following part of your input was truncated because CLIP can only handle sequences up to"
                        f" {tokenizer.model_max_length} tokens: {removed_text}"
                    )

                prompt_embeds = text_encoder(
                    text_input_ids.to(device),
                    output_hidden_states=True,
                )
                # We are only ALWAYS interested in the pooled output of the final text encoder
                pooled_prompt_embeds = prompt_embeds[0]

                prompt_embeds = prompt_embeds.hidden_states[-2]
                bs_embed, seq_len, _ = prompt_embeds.shape
                # duplicate text embeddings for each generation per prompt, using mps friendly method
                prompt_embeds = prompt_embeds.repeat(1, num_images_per_prompt, 1)
                prompt_embeds = prompt_embeds.view(bs_embed * num_images_per_prompt, seq_len, -1)
                prompt_embeds_list.append(prompt_embeds)

            prompt_embeds = torch.concat(prompt_embeds_list, dim=-1)
        # get unconditional embeddings for classifier free guidance
        zero_out_negative_prompt = negative_prompt is None and self.config.force_zeros_for_empty_prompt
        if do_classifier_free_guidance and negative_prompt_embeds is None and zero_out_negative_prompt:
            negative_prompt_embeds = torch.zeros_like(prompt_embeds)
            negative_pooled_prompt_embeds = torch.zeros_like(pooled_prompt_embeds)
        elif do_classifier_free_guidance and negative_prompt_embeds is None:
            negative_prompt = negative_prompt or ""
            uncond_tokens: List[str]
            if prompt is not None and type(prompt) is not type(negative_prompt):
                raise TypeError(
                    f"`negative_prompt` should be the same type to `prompt`, but got {type(negative_prompt)} !="
                    f" {type(prompt)}."
                )
            elif isinstance(negative_prompt, str):
                uncond_tokens = [negative_prompt]
            elif batch_size != len(negative_prompt):
                raise ValueError(
                    f"`negative_prompt`: {negative_prompt} has batch size {len(negative_prompt)}, but `prompt`:"
                    f" {prompt} has batch size {batch_size}. Please make sure that passed `negative_prompt` matches"
                    " the batch size of `prompt`."
                )
            else:
                uncond_tokens = negative_prompt

            negative_prompt_embeds_list = []
            for tokenizer, text_encoder in zip(tokenizers, text_encoders):
                # textual inversion: procecss multi-vector tokens if necessary
                if isinstance(self, TextualInversionLoaderMixin):
                    uncond_tokens = self.maybe_convert_prompt(uncond_tokens, tokenizer)

                max_length = prompt_embeds.shape[1]
                uncond_input = tokenizer(
                    uncond_tokens,
                    padding="max_length",
                    max_length=max_length,
                    truncation=True,
                    return_tensors="pt",
                )

                negative_prompt_embeds = text_encoder(
                    uncond_input.input_ids.to(device),
                    output_hidden_states=True,
                )
                # We are only ALWAYS interested in the pooled output of the final text encoder
                negative_pooled_prompt_embeds = negative_prompt_embeds[0]
                negative_prompt_embeds = negative_prompt_embeds.hidden_states[-2]

                if do_classifier_free_guidance:
                    # duplicate unconditional embeddings for each generation per prompt, using mps friendly method
                    seq_len = negative_prompt_embeds.shape[1]

                    negative_prompt_embeds = negative_prompt_embeds.to(dtype=text_encoder.dtype, device=device)

                    negative_prompt_embeds = negative_prompt_embeds.repeat(1, num_images_per_prompt, 1)
                    negative_prompt_embeds = negative_prompt_embeds.view(
                        batch_size * num_images_per_prompt, seq_len, -1
                    )

                    # For classifier free guidance, we need to do two forward passes.
                    # Here we concatenate the unconditional and text embeddings into a single batch
                    # to avoid doing two forward passes

                negative_prompt_embeds_list.append(negative_prompt_embeds)

            negative_prompt_embeds = torch.concat(negative_prompt_embeds_list, dim=-1)

        bs_embed = pooled_prompt_embeds.shape[0]
        pooled_prompt_embeds = pooled_prompt_embeds.repeat(1, num_images_per_prompt).view(
            bs_embed * num_images_per_prompt, -1
        )
        negative_pooled_prompt_embeds = negative_pooled_prompt_embeds.repeat(1, num_images_per_prompt).view(
            bs_embed * num_images_per_prompt, -1
        )
        return prompt_embeds, negative_prompt_embeds, pooled_prompt_embeds, negative_pooled_prompt_embeds

    # Copied from diffusers.pipelines.stable_diffusion.pipeline_stable_diffusion.StableDiffusionPipeline.prepare_extra_step_kwargs
    def prepare_extra_step_kwargs(self, generator, eta):
        # prepare extra kwargs for the scheduler step, since not all schedulers have the same signature
        # eta (η) is only used with the DDIMScheduler, it will be ignored for other schedulers.
        # eta corresponds to η in DDIM paper: https://arxiv.org/abs/2010.02502
        # and should be between [0, 1]

        accepts_eta = "eta" in set(inspect.signature(self.scheduler.step).parameters.keys())
        extra_step_kwargs = {}
        if accepts_eta:
            extra_step_kwargs["eta"] = eta

        # check if the scheduler accepts generator
        accepts_generator = "generator" in set(inspect.signature(self.scheduler.step).parameters.keys())
        if accepts_generator:
            extra_step_kwargs["generator"] = generator
        return extra_step_kwargs

    def check_inputs(
        self,
        prompt,
        height,
        width,
        callback_steps,
        negative_prompt=None,
        prompt_embeds=None,
        negative_prompt_embeds=None,
        pooled_prompt_embeds=None,
        negative_pooled_prompt_embeds=None,
    ):
        if height % 8 != 0 or width % 8 != 0:
            raise ValueError(f"`height` and `width` have to be divisible by 8 but are {height} and {width}.")

        if (callback_steps is None) or (
            callback_steps is not None and (not isinstance(callback_steps, int) or callback_steps <= 0)
        ):
            raise ValueError(
                f"`callback_steps` has to be a positive integer but is {callback_steps} of type"
                f" {type(callback_steps)}."
            )

        if prompt is not None and prompt_embeds is not None:
            raise ValueError(
                f"Cannot forward both `prompt`: {prompt} and `prompt_embeds`: {prompt_embeds}. Please make sure to"
                " only forward one of the two."
            )
        elif prompt is None and prompt_embeds is None:
            raise ValueError(
                "Provide either `prompt` or `prompt_embeds`. Cannot leave both `prompt` and `prompt_embeds` undefined."
            )
        elif prompt is not None and (not isinstance(prompt, str) and not isinstance(prompt, list)):
            raise ValueError(f"`prompt` has to be of type `str` or `list` but is {type(prompt)}")

        if negative_prompt is not None and negative_prompt_embeds is not None:
            raise ValueError(
                f"Cannot forward both `negative_prompt`: {negative_prompt} and `negative_prompt_embeds`:"
                f" {negative_prompt_embeds}. Please make sure to only forward one of the two."
            )

        if prompt_embeds is not None and negative_prompt_embeds is not None:
            if prompt_embeds.shape != negative_prompt_embeds.shape:
                raise ValueError(
                    "`prompt_embeds` and `negative_prompt_embeds` must have the same shape when passed directly, but"
                    f" got: `prompt_embeds` {prompt_embeds.shape} != `negative_prompt_embeds`"
                    f" {negative_prompt_embeds.shape}."
                )

        if prompt_embeds is not None and pooled_prompt_embeds is None:
            raise ValueError(
                "If `prompt_embeds` are provided, `pooled_prompt_embeds` also have to be passed. Make sure to generate `pooled_prompt_embeds` from the same text encoder that was used to generate `prompt_embeds`."
            )

        if negative_prompt_embeds is not None and negative_pooled_prompt_embeds is None:
            raise ValueError(
                "If `negative_prompt_embeds` are provided, `negative_pooled_prompt_embeds` also have to be passed. Make sure to generate `negative_pooled_prompt_embeds` from the same text encoder that was used to generate `negative_prompt_embeds`."
            )

    # Copied from diffusers.pipelines.stable_diffusion.pipeline_stable_diffusion.StableDiffusionPipeline.prepare_latents
    def prepare_latents(self, batch_size, num_channels_latents, height, width, dtype, device, generator, latents=None):
        shape = (batch_size, num_channels_latents, height // self.vae_scale_factor, width // self.vae_scale_factor)
        if isinstance(generator, list) and len(generator) != batch_size:
            raise ValueError(
                f"You have passed a list of generators of length {len(generator)}, but requested an effective batch"
                f" size of {batch_size}. Make sure the batch size matches the length of the generators."
            )

        if latents is None:
            latents = randn_tensor(shape, generator=generator, device=device, dtype=dtype)
        else:
            latents = latents.to(device)

        # scale the initial noise by the standard deviation required by the scheduler
        latents = latents * self.scheduler.init_noise_sigma
        return latents

    def _get_add_time_ids(self, original_size, crops_coords_top_left, target_size, dtype):
        add_time_ids = list(original_size + crops_coords_top_left + target_size)

        passed_add_embed_dim = (
            self.unet.config.addition_time_embed_dim * len(add_time_ids) + self.text_encoder_2.config.projection_dim
        )
        expected_add_embed_dim = self.unet.add_embedding.linear_1.in_features

        if expected_add_embed_dim != passed_add_embed_dim:
            raise ValueError(
                f"Model expects an added time embedding vector of length {expected_add_embed_dim}, but a vector of {passed_add_embed_dim} was created. The model has an incorrect config. Please check `unet.config.time_embedding_type` and `text_encoder_2.config.projection_dim`."
            )

        add_time_ids = torch.tensor([add_time_ids], dtype=dtype)
        return add_time_ids

    def load_img_f(self, path):
        image = Image.open(path).convert("RGB")
        w,h=1024,1024
        image = image.resize((w, h), resample=PIL.Image.LANCZOS)
        image = image.filter(ImageFilter.EDGE_ENHANCE_MORE)
        #image = image.filter(ImageFilter.CONTOUR)
        image = np.array(image).astype(np.float32) / 255.0
        image = image[None].transpose(0, 3, 1, 2)
        image = torch.from_numpy(image)
        return image #2.*image - 1.

    def load_img(self, path):
        image = Image.open(path).convert("RGB")
        w,h=1024,1024
        image = image.resize((w, h), resample=PIL.Image.LANCZOS)
        #image.save(path.split(".")[0]+"_input.png")
        image = image.filter(ImageFilter.CONTOUR)
        image = image.filter(ImageFilter.EDGE_ENHANCE_MORE)
        #image.save(path.split(".")[0]+"_input.png")
        image = np.array(image).astype(np.float32) / 255.0
        image = image[None].transpose(0, 3, 1, 2)
        image = torch.from_numpy(image)
        return 2.* image - 1.
        #return image
    
    def load_img_style(self, path):
        image = Image.open(path).convert("RGB")
        w,h=1024,1024
        image = image.resize((w, h), resample=PIL.Image.LANCZOS)
        #image.save(path.split(".")[0]+"_input.png")
        #image = image.filter(ImageFilter.CONTOUR)
        #image = image.filter(ImageFilter.EDGE_ENHANCE_MORE)
        image = np.array(image).astype(np.float32) / 255.0
        image = image[None].transpose(0, 3, 1, 2)
        image = torch.from_numpy(image)
        return image


    def prepare_pil2latent(
        self, image, timestep, batch_size, num_images_per_prompt, dtype, device, generator=None, add_noise=True
    ):
        if not isinstance(image, (torch.Tensor, PIL.Image.Image, list)):
            raise ValueError(
                f"`image` has to be of type `torch.Tensor`, `PIL.Image.Image` or list but is {type(image)}"
            )

        # Offload text encoder if `enable_model_cpu_offload` was enabled
        if hasattr(self, "final_offload_hook") and self.final_offload_hook is not None:
            self.text_encoder_2.to("cpu")
            torch.cuda.empty_cache()

        image = image.to(device=device, dtype=dtype)

        batch_size = batch_size * num_images_per_prompt

        if image.shape[1] == 4:
            init_latents = image

        else:
            # make sure the VAE is in float32 mode, as it overflows in float16
            if self.vae.config.force_upcast:
                image = image.to(dtype=torch.float32)
                self.vae.to(dtype=torch.float32)

            if isinstance(generator, list) and len(generator) != batch_size:
                raise ValueError(
                    f"You have passed a list of generators of length {len(generator)}, but requested an effective batch"
                    f" size of {batch_size}. Make sure the batch size matches the length of the generators."
                )

            elif isinstance(generator, list):
                init_latents = [
                    self.vae.encode(image[i : i + 1]).latent_dist.sample(generator[i]) for i in range(batch_size)
                ]
                init_latents = torch.cat(init_latents, dim=0)
            else:
                init_latents = self.vae.encode(image).latent_dist.sample(generator)

            if self.vae.config.force_upcast:
                self.vae.to(dtype)

            init_latents = init_latents.to(dtype)
            init_latents = self.vae.config.scaling_factor * init_latents

        if batch_size > init_latents.shape[0] and batch_size % init_latents.shape[0] == 0:
            # expand init_latents for batch_size
            additional_image_per_prompt = batch_size // init_latents.shape[0]
            init_latents = torch.cat([init_latents] * additional_image_per_prompt, dim=0)
        elif batch_size > init_latents.shape[0] and batch_size % init_latents.shape[0] != 0:
            raise ValueError(
                f"Cannot duplicate `image` of batch size {init_latents.shape[0]} to {batch_size} text prompts."
            )
        else:
            init_latents = torch.cat([init_latents], dim=0)
        

        if add_noise:
            shape = init_latents.shape
            noise = randn_tensor(shape, generator=generator, device=device, dtype=dtype)
            #timestep=torch.full(shape,timestep).cuda()
            # get latents
            init_latents = self.scheduler.add_noise(init_latents, noise, timestep)

        latents = init_latents

        return latents

    def ddim_next_step(
            self,
            model_output: torch.FloatTensor,  # 对应 noise_pred
            timestep: int,
            x: torch.FloatTensor,  # 对应 latents
            eta=0.0,
            verbose=False,
    ):
        """
        Inverse sampling for DDIM Inversion  （x_t-1 -> x_t
        """
        if verbose:
            print("timestep: ", timestep)
        next_step = timestep
        timestep = min(
            timestep
            - self.scheduler.config.num_train_timesteps
            // self.scheduler.num_inference_steps,
            999,
            )
        alpha_prod_t = (
            self.scheduler.alphas_cumprod[timestep]
            if timestep >= 0
            else self.scheduler.final_alpha_cumprod # 第一次timestpe=19,alpha_prod_t取alphas_cumprod[0]
        )  # \bar_{\alpha}_t
        alpha_prod_t_next = self.scheduler.alphas_cumprod[next_step] #
        # beta_prod_t = 1 - alpha_prod_t  #

        # 根据x_t计算出x_0 :  x_0 = {x_t - sqrt(1- \bar_{\alpha_t})*model_output } / {sqrt(\bar_{\alpha_t})}
        pred_x0 = (x - (1 - alpha_prod_t)**0.5 * model_output) / alpha_prod_t**0.5

        # pred_dir = (1 - alpha_prod_t_next) ** 0.5 * model_output
        # 再依据x_0计算出 x_(t+1) : x_t+1 = sqrt( \bar_{\alpha_t}) * x_0 + \sqrt(1- \bar_{\alpha_t}) * model_output
        x_next = alpha_prod_t_next**0.5 * pred_x0 + (1 - alpha_prod_t_next) ** 0.5 * model_output
        return x_next, pred_x0

    def ddim_invert(
            self,
            image,
            # prompt,
            num_inference_steps=50,
            guidance_scale = 5.0,
            encoder_hidden_states=None,
            cross_attention_kwargs=None,
            added_cond_kwargs=None,
            eta = 0.0,
            return_intermediates = False ,
    ):
        """
        invert a real image into noise map with determinisc DDIM inversion
        """
        DEVICE = (
            torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
        )
        batch_size = image.shape[0]

        # text embeddings

        # define initial latents
        image = image.to(self.device)
        latents = self.prepare_pil2latent(image,0,1,1,
                                          torch.float16,image.device,add_noise=False).to(self.unet.dtype)
        start_latents = latents

        # interative sampling
        self.scheduler.set_timesteps(num_inference_steps) # 设置去噪步长:  t in [981,961,···21,1]
        print("Valid timesteps: ", reversed(self.scheduler.timesteps)) # print倒转的t值  倒转后是[1,21,···,961,981]
        # print("attributes: ", self.scheduler.__dict__)
        latents_dict = dict()
        pred_x0_dict = dict()
        latents_dict[-1] = [latents] # x0
        pred_x0_dict[-1] = [latents] # x0
        for i, t in enumerate(
                tqdm(reversed(self.scheduler.timesteps), desc="DDIM Inversion")  # t in [1,21,41···,961,981]
        ):
            if guidance_scale > 1.0:
                model_inputs = torch.cat([latents] * 2) # (2,4,64,64)
            else:
                model_inputs = latents

            # predict the noise
            noise_pred,_ = self.unet(
                model_inputs, t,  encoder_hidden_states=encoder_hidden_states,
                cross_attention_kwargs=cross_attention_kwargs,
                added_cond_kwargs=added_cond_kwargs,
                return_dict=False,
            ) # 得到 noise_pred 和 unet各阶段的feat
            noise_pred = noise_pred[0]

            if guidance_scale > 1.0:
                noise_pred_uncon, noise_pred_con = noise_pred.chunk(2, dim=0)
                noise_pred = noise_pred_uncon + guidance_scale * (
                        noise_pred_con - noise_pred_uncon
                )
            # compute the previous noise sample x_t-1 -> x_t : x0->x1->x21->x41 ···->x982  [t的取值是跳跃的]
            latents, pred_x0 = self.ddim_next_step(noise_pred, t, latents)  # 得到 x_t 和 由x_t-1计算得到x_0
            if batch_size == 1 :
                latents_dict[int(t)] = latents  # 得到 x0 x1 x21 x41 ··· x981
                pred_x0_dict[int(t)] = pred_x0  # 1个真实的x0和 50个由{x0,x1,x21,x41·· x961}预测的x0
            elif batch_size == 2 :
                _,__ = latents.chunk(2)
                latents_dict[int(t)] = _
                _,__ = pred_x0.chunk(2)
                pred_x0_dict[int(t)] = _

        if return_intermediates:
            # return the intermediate laters during inversion
            # pred_x0_list = [self.latent2image(img, return_type="pt") for img in pred_x0_list]
            return latents, latents_dict, pred_x0_dict
        return latents, start_latents, latents_dict, pred_x0_dict




    @torch.no_grad()
    @replace_example_docstring(EXAMPLE_DOC_STRING)
    def __call__(
        self,
        prompt: Union[str, List[str]] = None,
        height: Optional[int] = None,
        width: Optional[int] = None,
        num_inference_steps: int = 50,
        guidance_scale: float = 5.0,
        negative_prompt: Optional[Union[str, List[str]]] = None,
        num_images_per_prompt: Optional[int] = 1,
        eta: float = 0.0,
        generator: Optional[Union[torch.Generator, List[torch.Generator]]] = None,
        latents: Optional[torch.FloatTensor] = None,
        prompt_embeds: Optional[torch.FloatTensor] = None,
        negative_prompt_embeds: Optional[torch.FloatTensor] = None,
        pooled_prompt_embeds: Optional[torch.FloatTensor] = None,
        negative_pooled_prompt_embeds: Optional[torch.FloatTensor] = None,
        output_type: Optional[str] = "pil",
        return_dict: bool = True,
        callback: Optional[Callable[[int, int, torch.FloatTensor], None]] = None,
        callback_steps: int = 1,
        cross_attention_kwargs: Optional[Dict[str, Any]] = None,
        guidance_rescale: float = 0.0,
        original_size: Optional[Tuple[int, int]] = None,
        crops_coords_top_left: Tuple[int, int] = (0, 0),
        target_size: Optional[Tuple[int, int]] = None,
        refimgpath=None,
        styleimgpath=None,
        x0_img_vis = False,
        replace_phase_after_timestep= None,
        is_adain_during_replace_phase = False,
        cnt_guidance_timestep = None,
        ddim_inver = False,
        replace_phase_in_unet = None,
        adain_with_patches = False,
        use_cnt_mid_feat_in_unet = False,
        is_freestyle = False,
    ):
        r"""
        Function invoked when calling the pipeline for generation.

        Args:
            prompt (`str` or `List[str]`, *optional*):
                The prompt or prompts to guide the image generation. If not defined, one has to pass `prompt_embeds`.
                instead.
            height (`int`, *optional*, defaults to self.unet.config.sample_size * self.vae_scale_factor):
                The height in pixels of the generated image.
            width (`int`, *optional*, defaults to self.unet.config.sample_size * self.vae_scale_factor):
                The width in pixels of the generated image.
            num_inference_steps (`int`, *optional*, defaults to 50):
                The number of denoising steps. More denoising steps usually lead to a higher quality image at the
                expense of slower inference.
            guidance_scale (`float`, *optional*, defaults to 7.5):
                Guidance scale as defined in [Classifier-Free Diffusion Guidance](https://arxiv.org/abs/2207.12598).
                `guidance_scale` is defined as `w` of equation 2. of [Imagen
                Paper](https://arxiv.org/pdf/2205.11487.pdf). Guidance scale is enabled by setting `guidance_scale >
                1`. Higher guidance scale encourages to generate images that are closely linked to the text `prompt`,
                usually at the expense of lower image quality.
            negative_prompt (`str` or `List[str]`, *optional*):
                The prompt or prompts not to guide the image generation. If not defined, one has to pass
                `negative_prompt_embeds` instead. Ignored when not using guidance (i.e., ignored if `guidance_scale` is
                less than `1`).
            num_images_per_prompt (`int`, *optional*, defaults to 1):
                The number of images to generate per prompt.
            eta (`float`, *optional*, defaults to 0.0):
                Corresponds to parameter eta (η) in the DDIM paper: https://arxiv.org/abs/2010.02502. Only applies to
                [`schedulers.DDIMScheduler`], will be ignored for others.
            generator (`torch.Generator` or `List[torch.Generator]`, *optional*):
                One or a list of [torch generator(s)](https://pytorch.org/docs/stable/generated/torch.Generator.html)
                to make generation deterministic.
            latents (`torch.FloatTensor`, *optional*):
                Pre-generated noisy latents, sampled from a Gaussian distribution, to be used as inputs for image
                generation. Can be used to tweak the same generation with different prompts. If not provided, a latents
                tensor will ge generated by sampling using the supplied random `generator`.
            prompt_embeds (`torch.FloatTensor`, *optional*):
                Pre-generated text embeddings. Can be used to easily tweak text inputs, *e.g.* prompt weighting. If not
                provided, text embeddings will be generated from `prompt` input argument.
            negative_prompt_embeds (`torch.FloatTensor`, *optional*):
                Pre-generated negative text embeddings. Can be used to easily tweak text inputs, *e.g.* prompt
                weighting. If not provided, negative_prompt_embeds will be generated from `negative_prompt` input
                argument.
            pooled_prompt_embeds (`torch.FloatTensor`, *optional*):
                Pre-generated pooled text embeddings. Can be used to easily tweak text inputs, *e.g.* prompt weighting.
                If not provided, pooled text embeddings will be generated from `prompt` input argument.
            negative_pooled_prompt_embeds (`torch.FloatTensor`, *optional*):
                Pre-generated negative pooled text embeddings. Can be used to easily tweak text inputs, *e.g.* prompt
                weighting. If not provided, pooled negative_prompt_embeds will be generated from `negative_prompt`
                input argument.
            output_type (`str`, *optional*, defaults to `"pil"`):
                The output format of the generate image. Choose between
                [PIL](https://pillow.readthedocs.io/en/stable/): `PIL.Image.Image` or `np.array`.
            return_dict (`bool`, *optional*, defaults to `True`):
                Whether or not to return a [`~pipelines.stable_diffusion.StableDiffusionXLPipelineOutput`] instead of a
                plain tuple.
            callback (`Callable`, *optional*):
                A function that will be called every `callback_steps` steps during inference. The function will be
                called with the following arguments: `callback(step: int, timestep: int, latents: torch.FloatTensor)`.
            callback_steps (`int`, *optional*, defaults to 1):
                The frequency at which the `callback` function will be called. If not specified, the callback will be
                called at every step.
            cross_attention_kwargs (`dict`, *optional*):
                A kwargs dictionary that if specified is passed along to the `AttentionProcessor` as defined under
                `self.processor` in
                [diffusers.cross_attention](https://github.com/huggingface/diffusers/blob/main/src/diffusers/models/cross_attention.py).
            guidance_rescale (`float`, *optional*, defaults to 0.7):
                Guidance rescale factor proposed by [Common Diffusion Noise Schedules and Sample Steps are
                Flawed](https://arxiv.org/pdf/2305.08891.pdf) `guidance_scale` is defined as `φ` in equation 16. of
                [Common Diffusion Noise Schedules and Sample Steps are Flawed](https://arxiv.org/pdf/2305.08891.pdf).
                Guidance rescale factor should fix overexposure when using zero terminal SNR.
            original_size (`Tuple[int]`, *optional*, defaults to (1024, 1024)):
                TODO
            crops_coords_top_left (`Tuple[int]`, *optional*, defaults to (0, 0)):
                TODO
            target_size (`Tuple[int]`, *optional*, defaults to (1024, 1024)):
                TODO

        Examples:

        Returns:
            [`~pipelines.stable_diffusion.StableDiffusionXLPipelineOutput`] or `tuple`:
            [`~pipelines.stable_diffusion.StableDiffusionXLPipelineOutput`] if `return_dict` is True, otherwise a
            `tuple. When returning a tuple, the first element is a list with the generated images, and the second
            element is a list of `bool`s denoting whether the corresponding generated image likely represents
            "not-safe-for-work" (nsfw) content, according to the `safety_checker`.
        """
        # 0. Default height and width to unet
        height = height or self.default_sample_size * self.vae_scale_factor
        width = width or self.default_sample_size * self.vae_scale_factor

        original_size = original_size or (height, width)
        target_size = target_size or (height, width)
        self.adain_with_patches = adain_with_patches

        # 1. Check inputs. Raise error if not correct
        self.check_inputs(
            prompt,
            height,
            width,
            callback_steps,
            negative_prompt,
            prompt_embeds,
            negative_prompt_embeds,
            pooled_prompt_embeds,
            negative_pooled_prompt_embeds,
        )

        # 2. Define call parameters
        if prompt is not None and isinstance(prompt, str):
            batch_size = 1
        elif prompt is not None and isinstance(prompt, list):
            batch_size = len(prompt)
        else:
            batch_size = prompt_embeds.shape[0]

        device = self._execution_device

        # here `guidance_scale` is defined analog to the guidance weight `w` of equation (2)
        # of the Imagen paper: https://arxiv.org/pdf/2205.11487.pdf . `guidance_scale = 1`
        # corresponds to doing no classifier free guidance.
        do_classifier_free_guidance = guidance_scale > 1.0

        # 3. Encode input prompt
        text_encoder_lora_scale = (
            cross_attention_kwargs.get("scale", None) if cross_attention_kwargs is not None else None
        )
        print("text:",prompt)
        (
            prompt_embeds,
            negative_prompt_embeds,
            pooled_prompt_embeds,
            negative_pooled_prompt_embeds,
        ) = self.encode_prompt(
            prompt,
            device,
            num_images_per_prompt,
            do_classifier_free_guidance,
            negative_prompt,
            prompt_embeds=prompt_embeds,
            negative_prompt_embeds=negative_prompt_embeds,
            pooled_prompt_embeds=pooled_prompt_embeds,
            negative_pooled_prompt_embeds=negative_pooled_prompt_embeds,
            lora_scale=text_encoder_lora_scale,
        )
        (
            Nullprompt_embeds,
            Nullnegative_prompt_embeds,
            Nullpooled_prompt_embeds,
            Nullnegative_pooled_prompt_embeds,
        ) = self.encode_prompt(
            "",
            device,
            num_images_per_prompt,
            do_classifier_free_guidance,
            negative_prompt,
        )
        # 4. Prepare timesteps
        self.scheduler.set_timesteps(num_inference_steps, device=device)

        timesteps = self.scheduler.timesteps
        #Prepare ref img
        
        print("the style image path is:",styleimgpath)
        imgprompt=None
        if self.refimg==None and refimgpath!=None:
            self.refimg = self.load_img(refimgpath)#.to(device)
            self.refimg = self.image_processor.preprocess(self.refimg)
            self.refimg_f = self.load_img_f(refimgpath)#.to(device)
            self.refimg_f = self.image_processor.preprocess(self.refimg_f)
            
            self.styleimg=None
            
        # 5. Prepare latent variables
        num_channels_latents = self.unet.config.in_channels
        #code added
        latents = self.prepare_pil2latent(self.refimg_f, # RGB-tensor [1,3,1024,1024]
                                            torch.tensor([958]).repeat(batch_size * num_images_per_prompt).to(device),
                                            batch_size,
                                            num_images_per_prompt,
                                            prompt_embeds.dtype,
                                            device,
                                            generator,
                                            add_noise=True,
                                            )

        if self.styleimg!=None:
            imgprompt=imgprompt.unsqueeze(0)
            imgprompt=imgprompt.repeat(num_images_per_prompt,77,1)
            prompt_temp=Nullprompt_embeds
            prompt_temp=imgprompt
            prompt_embeds=prompt_temp.half().cuda()
            pooled_prompt_embeds=imgprompt_2.repeat(num_images_per_prompt,1).half().cuda()
       
        """
            latents = self.prepare_pil2latent(self.styleimg,
                                            torch.tensor([958]).repeat(batch_size * num_images_per_prompt).to(device),
                                            batch_size,
                                            num_images_per_prompt,
                                            prompt_embeds.dtype,
                                            device,
                                            generator,
                                            add_noise=True,
                                            )

        """

        # 6. Prepare extra step kwargs. TODO: Logic should ideally just be moved out of the pipeline
        extra_step_kwargs = self.prepare_extra_step_kwargs(generator, eta)

        # 7. Prepare added time ids & embeddings
        add_text_embeds = pooled_prompt_embeds
        add_time_ids = self._get_add_time_ids(
            original_size, crops_coords_top_left, target_size, dtype=prompt_embeds.dtype
        )

        if do_classifier_free_guidance:
            ddim_nullprompt_embeds = torch.cat([Nullprompt_embeds]*2,dim=0)
            prompt_embeds = torch.cat([negative_prompt_embeds, prompt_embeds], dim=0)
            Nullprompt_embeds = torch.cat([Nullnegative_prompt_embeds, Nullprompt_embeds], dim=0)
            add_text_embeds = torch.cat([negative_pooled_prompt_embeds, add_text_embeds], dim=0)
            add_time_ids = torch.cat([add_time_ids, add_time_ids], dim=0)

        ddim_nullprompt_embeds = ddim_nullprompt_embeds.to(device)
        prompt_embeds = prompt_embeds.to(device)
        Nullprompt_embeds=Nullprompt_embeds.to(device)
        add_text_embeds = add_text_embeds.to(device)
        add_time_ids = add_time_ids.to(device).repeat(batch_size * num_images_per_prompt, 1)

        added_cond_kwargs = {"text_embeds": add_text_embeds, "time_ids": add_time_ids}

        # #  DDIM inversion
        self.unet.set_branch(False)
        if ddim_inver : # not use 
            _ , latents_list, x0pred_list = self.ddim_invert(  # DDIM Inversion add noise 
                self.refimg_f,
                num_inference_steps = num_inference_steps,
                guidance_scale = guidance_scale,
                encoder_hidden_states= Nullprompt_embeds , # prompt_embeds ,# Nullprompt_embeds, # Nullprompt_embeds,
                cross_attention_kwargs=cross_attention_kwargs,
                added_cond_kwargs=added_cond_kwargs,
                return_intermediates=True,
            )  # obtain x981 & latent_list{x0 x1 x21 x41 ··· x981}
        # self.unet.set_branch(True)
        # self.unet.set_replace_phase_in_unet(replace_phase_in_unet)

        # 8. Denoising loop
        num_warmup_steps = len(timesteps) - num_inference_steps * self.scheduler.order
        """latents = self.prepare_pil2latent(self.refimg,
                                            torch.tensor([298.]).repeat(batch_size * num_images_per_prompt).to(device),
                                            batch_size,
                                            num_images_per_prompt,
                                            prompt_embeds.dtype,
                                            device,
                                            generator,
                                            add_noise=True,
                                            )"""

        # self.register_cross_attention_inUpBlock(self.unet)  # register ca

        with self.progress_bar(total=num_inference_steps) as progress_bar:
            for i, t in enumerate(timesteps):
                self.t = i
                #print(t)
                #if t>299:
                #    continue
                skip_stack=None
                self.unet.set_curr_index(i+1)
                if self.refimg!=None:
                    """
                    if t-30>0:
                        print("t-30>0")
                        t_pre=t-30
                    """
                    cnt_latents = None
                    if ddim_inver:
                        cnt_latents = latents_list[int(t)]
                        cnt_latents = torch.cat([cnt_latents] * 2) if do_classifier_free_guidance else cnt_latents
                    reflatents = self.prepare_pil2latent(self.refimg_f,  # content branch
                                                        torch.tensor([0]).repeat(batch_size * num_images_per_prompt).to(device),
                                                        batch_size,
                                                        num_images_per_prompt,
                                                        prompt_embeds.dtype,
                                                        device,
                                                        generator,
                                                        add_noise=False,
                                                        )
                    reflatent_model_input = torch.cat([reflatents] * 2) if do_classifier_free_guidance else reflatents # content branch
                    reflatent_model_input = self.scheduler.scale_model_input(reflatent_model_input, t)
                # expand the latents if we are doing classifier free guidance
                # print(f"{i} latents.mean: {latents.mean()},latents.var: {latents.var()} ")
                latent_model_input = torch.cat([latents] * 2) if do_classifier_free_guidance else latents
                latent_model_input = self.scheduler.scale_model_input(latent_model_input, t)

                cnt_mid_feat = None
                if use_cnt_mid_feat_in_unet :
                    cnt_mid_feat = self.unet(
                        reflatent_model_input,         # reflatent_model_input,  content image
                        t,
                        # encoder_hidden_states= Nullprompt_embeds,
                        encoder_hidden_states= prompt_embeds, # prompt_embeds,
                        cross_attention_kwargs=cross_attention_kwargs,
                        added_cond_kwargs=added_cond_kwargs,
                        return_dict=False,
                        #refsample=reflatent_model_input,
                        skip_stack=None,
                        return_cnt_mid_feat = True,
                        cnt_mid_feat = None,
                    )

                # # predict the noise residual
                noise_pred, skip_stack = self.unet(
                    latent_model_input,   #  noisy_img    
                    #reflatent_model_input,
                    t,
                    encoder_hidden_states = prompt_embeds,   # prompt_embeds,
                    cross_attention_kwargs=cross_attention_kwargs,
                    added_cond_kwargs=added_cond_kwargs,
                    return_dict=False ,
                    #refsample=reflatent_model_input,
                    skip_stack= None ,
                    return_skip_stack=False ,  # True: pass Unet's decoder，False: go thourgh all unet 
                    return_cnt_mid_feat = False,
                    cnt_mid_feat = cnt_mid_feat,
                )

                noise_pred=noise_pred[0]

                # perform guidance
                if do_classifier_free_guidance:
                    noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
                    noise_pred = noise_pred_uncond + guidance_scale * (noise_pred_text - noise_pred_uncond)

                if do_classifier_free_guidance and guidance_rescale > 0.0:
                    # Based on 3.4. in https://arxiv.org/pdf/2305.08891.pdf
                    noise_pred = rescale_noise_cfg(noise_pred, noise_pred_text, guidance_rescale=guidance_rescale)

                # # # cnt guidance
                if cnt_guidance_timestep is not None and i in cnt_guidance_timestep:
                    self.unet.set_branch(False)
                    cnt_pred,skip_stack = self.unet(
                        cnt_latents,
                        t,
                        encoder_hidden_states = prompt_embeds, # Nullprompt_embeds
                        cross_attention_kwargs = cross_attention_kwargs,
                        added_cond_kwargs = added_cond_kwargs,
                        return_dict=False,
                        skip_stack=None,
                        return_skip_stack= False ,
                    )
                    self.unet.set_branch(True)
                    cnt_pred = cnt_pred[0]
                    if do_classifier_free_guidance:
                        cnt_pred_uncond, cnt_pred_text = cnt_pred.chunk(2)
                        cnt_pred = cnt_pred_uncond + guidance_scale * (cnt_pred_text - cnt_pred_uncond)

                    w = 1.3
                    noise_pred = cnt_pred + w * (noise_pred - cnt_pred)

                # print(latents.shape, noise_pred.shape)
                # compute the previous noisy sample x_t -> x_t-1
                if replace_phase_after_timestep is None:
                    # # FreeStyle official
                    # latents = self.scheduler.step(noise_pred, t, latents, **extra_step_kwargs, return_dict=False)[0]
                    latents = self.update_latents(noise_pred, t, latents,
                                                  t_index=i ,
                                                  refimg_f = self.refimg_f,
                                                  x0_img_vis = x0_img_vis,
                                                  replace_phase_after_timestep=replace_phase_after_timestep,
                                                  return_dict=False,
                                                  )[0]

                else :
                    # our Phase Diffusion 
                    latents = self.update_latents(noise_pred, t, latents,
                                                      t_index=i ,
                                                      refimg_f = self.refimg_f,
                                                      x0_img_vis = x0_img_vis,
                                                      is_adain = is_adain_during_replace_phase,
                                                      replace_phase_after_timestep=replace_phase_after_timestep,
                                                      return_dict=False,
                                                      prompt_embeds = prompt_embeds,   # prompt_embeds,
                                                      cross_attention_kwargs=cross_attention_kwargs,
                                                      added_cond_kwargs=added_cond_kwargs,
                                                      classifier_guidance = guidance_scale ,
                                                      )[0]

                # call the callback, if provided
                if i == len(timesteps) - 1 or ((i + 1) > num_warmup_steps and (i + 1) % self.scheduler.order == 0):
                    progress_bar.update()
                    if callback is not None and i % callback_steps == 0:
                        callback(i, t, latents)

        self.refimg=None
        # make sure the VAE is in float32 mode, as it overflows in float16
        self.vae.to(dtype=torch.float32)

        use_torch_2_0_or_xformers = self.vae.decoder.mid_block.attentions[0].processor in [
            AttnProcessor2_0,
            XFormersAttnProcessor,
            LoRAXFormersAttnProcessor,
            LoRAAttnProcessor2_0,
        ]
        # if xformers or torch_2_0 is used attention block does not need
        # to be in float32 which can save lots of memory
        if not use_torch_2_0_or_xformers:
            self.vae.post_quant_conv.to(latents.dtype)
            self.vae.decoder.conv_in.to(latents.dtype)
            self.vae.decoder.mid_block.to(latents.dtype)
        else:
            latents = latents.float()

        if not output_type == "latent":
            image = self.vae.decode(latents / self.vae.config.scaling_factor, return_dict=False)[0]
        else:
            image = latents
            return StableDiffusionXLPipelineOutput(images=image)

        image = self.image_processor.postprocess(image, output_type=output_type)

        # Offload last model to CPU
        if hasattr(self, "final_offload_hook") and self.final_offload_hook is not None:
            self.final_offload_hook.offload()

        if not return_dict:
            return (image,)

        return StableDiffusionXLPipelineOutput(images=image)


    def update_latents(
            self,
            model_output,
            timestep,
            latent,
            eta=0.0,
            generator=None,
            variance_noise = None,
            return_dict = True,
            replace_phase_after_timestep = None,
            model_output_vis = False,
            refimg_f = None,  # RGB
            x0_img_vis = False,
            t_index = None,
            record_x0_pred = False,
            is_adain = True,
            prompt_embeds = None,   # prompt_embeds,
            cross_attention_kwargs=None,
            added_cond_kwargs=None,
            classifier_guidance = None,
            return_x0pred = False,
    ):

        if replace_phase_after_timestep:
            if not isinstance(replace_phase_after_timestep,list):
                replace_phase_after_timestep = [replace_phase_after_timestep]

        # 1. get previous step value (=t-1)
        prev_timestep = timestep - self.scheduler.config.num_train_timesteps // self.scheduler.num_inference_steps

        # 2. compute alphas, betas
        alpha_prod_t = self.scheduler.alphas_cumprod[timestep]
        alpha_prod_t_prev = self.scheduler.alphas_cumprod[prev_timestep] if prev_timestep >= 0 else self.scheduler.final_alpha_cumprod

        beta_prod_t = 1 - alpha_prod_t

        dtype = model_output.dtype
        device = model_output.device

        # 3. compute predicted x0 from predicted noise also called
        pred_original_sample = (latent - beta_prod_t ** (0.5) * model_output) / alpha_prod_t ** (0.5)  # caculate x_0,  according to x_t
        if return_x0pred :
            return pred_original_sample

        if record_x0_pred:
            self.x0_pred_dict[t_index] = pred_original_sample

        # # 4. FFT feature
        # 4.1 RGB space
        if replace_phase_after_timestep and t_index in replace_phase_after_timestep and not record_x0_pred :
            old_pred_original_sample = pred_original_sample
            x0 = self.latent_to_img_via_vaeDecoder(pred_original_sample,'pt').to(device) #  +1 *255 not in this function
            ref_sty = x0.detach()  # 1024x1024
            old_x0pred = x0
            # x0 [1,3,1024,1024]
            if not x0.dtype == torch.float32:
                x0 = x0.to(dtype=torch.float32)

            # # content image 
            ref_img = refimg_f.to(device)
            if not ref_img.dtype == torch.float32:
                ref_img.to(dtype=torch.float32)

            latents = None
            print(f" replace x0 phase on timestep {t_index} ")
            # latents = x0
            x0_max = x0.max()
            x0_min = x0.min()
            # x0 = (x0 - x0_min) / (x0_max - x0_min)
            ref_cnt = ref_img
            ref_img = (ref_img - ref_img.min()) / (ref_img.max() - ref_img.min())

            x0_frequency = torch.fft.fft2(x0,dim=(2,3))
            content_frequency = torch.fft.fft2(ref_img,dim=(2,3))

            content_m = torch.abs(content_frequency)
            content_p = torch.angle(content_frequency)
            # alpha = 0.8
            # x0_frequency = x0_frequency * alpha + content_frequency * (1 - alpha)
            x0_m = torch.abs(x0_frequency)

            latents_freq = (x0_m)  * torch.exp(1j*content_p)  # amplitude  * torch.exp(1j*phase)
            # latents_freq = latents_freq * 1.1
            latents = torch.abs(torch.fft.ifft2(latents_freq,dim=(2,3)).real).to(dtype)

            latents = latents * (x0_max-x0_min) + x0_min  #  [b,c,h,w]

            # # patches-adain
            if self.adain_with_patches:
                patch_size = 256
                content_patches_list = self.cut_tensor(latents, patch_size) # 【1，3，256，64，64】
                style_patches_list = self.cut_tensor(ref_sty, patch_size)

                # 应用 AdaIN 操作
                content_patches_list = [ self.adain(content_patch,sty_patch) for content_patch, sty_patch in zip(content_patches_list,style_patches_list)]

                latents = self.patches_to_image(content_patches_list, patch_size)
                self.adain_with_patches = False
            else:
                latents = self.adain(latents, ref_sty)

            pred_original_sample = self.img_latent_via_vaeEncoder(latents.to(device=self.device)) # x0_hat

            pred_original_sample = pred_original_sample.detach()
            pred_original_sample = pred_original_sample * 0.5 + 0.7 * old_pred_original_sample

        pred_original_sample = pred_original_sample.to(dtype)
        if x0_img_vis :
            self.x0_vis(pred_original_sample,t_index)
        if model_output_vis :
            self.model_output_vis(model_output,timestep)

        # 5. compute variance: "sigma_t(η)" -> see formula (16)
        # σ_t = sqrt((1 − α_t−1)/(1 − α_t)) * sqrt(1 − α_t/α_t−1)
        variance = self.scheduler._get_variance(timestep, prev_timestep)
        std_dev_t = eta * variance ** (0.5)

        # 6. compute "direction pointing to x_t" of formula (12) from https://arxiv.org/pdf/2010.02502.pdf
        pred_sample_direction = (1 - alpha_prod_t_prev - std_dev_t**2) ** (0.5) * model_output

        # 7. compute x_t without "random noise" of formula (12) from https://arxiv.org/pdf/2010.02502.pdf
        prev_sample = alpha_prod_t_prev ** (0.5) * pred_original_sample + pred_sample_direction  # caculate xt_1, according to x0_t

        if not return_dict:
            return (prev_sample.to(dtype),)  # 返回xt-1

    def register_cross_attention_inUpBlock(self,unet):
        from diffusers.models import attention_processor
        import torch.nn as nn

        class DefaultAttentionProcessor(nn.Module):
            def __init__(self):
                super().__init__()
                self.processor = attention_processor.AttnProcessor2_0()

            def __call__(self, attn: attention_processor.Attention, hidden_states, encoder_hidden_states=None,
                         attention_mask=None, **kwargs):
                return self.processor(attn, hidden_states, encoder_hidden_states, attention_mask)
        class AttentionProcessor :
            def __init__(self):
                pass

            def __call__(self,
                         attn,
                         x,
                         encoder_hidden_states=None,
                         attention_mask=None,
                         context=None,temb=None):
                """
                   The attention is similar to the original implementation of LDM CrossAttention class
                   except adding some modifications on the attention
                """
                residual = x

                if attn.spatial_norm is not None:
                    x = attn.spatial_norm(x,temb)

                input_ndim = x.ndim

                if input_ndim == 4:
                    batch_size, channel, height, width = x.shape
                    x = x.view(batch_size, channel, height*width).transpose(1,2)

                batch_size, sequence_length, _ = (
                    x.shape if encoder_hidden_states is None else encoder_hidden_states.shape
                )

                if attention_mask is not None:
                    attention_mask = attn.prepare_attention_mask(attention_mask, sequence_length, batch_size)
                    attention_mask = attention_mask.view(batch_size, attn.heads, -1, attention_mask.shape[-1])

                if attn.group_norm is not None:
                    x = attn.group_norm(x.transpose(1,2)).transpose(1,2)

                query = attn.to_q(x) # [4,a,b]

                is_cross = encoder_hidden_states is not None
                if not is_cross:
                    encoder_hidden_states = x
                elif attn.norm_cross:
                    encoder_hidden_states = attn.norm_encoder_hidden_states(encoder_hidden_states)

                key = attn.to_k(encoder_hidden_states)
                value = attn.to_v(encoder_hidden_states)  # [2,77,1280]

                inner_dim = key.shape[-1]
                head_dim = inner_dim // attn.heads

                query = query.view(batch_size, -1 , attn.heads, head_dim).transpose(1,2)

                key = key.view(batch_size, -1, attn.heads, head_dim).transpose(1,2)
                value = value.view(batch_size, -1, attn.heads, head_dim).transpose(1,2) # [2,20,77,64]

                out = compute_scaled_dot_product_attention(
                    query, key, value, attn_mask=attention_mask,is_cross=is_cross,
                )  # [2,20,1024,64]

                out = out.transpose(1,2).reshape(batch_size, -1, attn.heads*head_dim) # [2,1024,1280]
                out = out.to(query.dtype)

                # linear
                out = attn.to_out[0](out)
                # dropout
                out = attn.to_out[1](out)

                if input_ndim == 4 :
                    out = out.transpose(-1,2).reshape(batch_size, channel, height, width)

                if attn.residual_connection:
                    out = out + residual

                out = out/attn.rescale_output_factor

                return out
        def compute_scaled_dot_product_attention(Q, K, V,attn_mask = None, is_cross=False, contrast_strength=1.0):
            """ Compute the scale dot product attention, potentially with our contrasting operation. """ # contrast_strength enlarge variance

            attn_weight = torch.softmax((Q @ K.transpose(-2, -1) / math.sqrt(Q.size(-1))), dim=-1)
            def save_heatmap(weights, batch_idx, save_path='attention_heatmap.png'):
                
                weights_2d_avg = weights.mean(dim=(0, 1)).detach().cpu().numpy()

                # 生成heatmap
                sns.heatmap(weights_2d_avg, cmap='viridis')
                plt.title("Average Attention Weights across Batch and Heads")
                plt.xlabel('Key Positions')
                plt.ylabel('Query Positions')
                plt.show()

                plt.figure(figsize=(10, 8))
                sns.heatmap(weights[batch_idx].detach().cpu().numpy(), cmap='viridis')
                plt.title('Attention Weights Heatmap')
                plt.xlabel('Key')
                plt.ylabel('Query')
                plt.savefig(save_path)
                plt.close()

            # save first heatmap
            # save_heatmap(attn_weight, batch_idx=1)
            def enhance_tensor(tensor: torch.Tensor, contrast_factor: float = 1.67) -> torch.Tensor:
                """ Compute the attention map contrasting. """
                adjusted_tensor = (tensor - tensor.mean(dim=-1)) * contrast_factor + tensor.mean(dim=-1)  # enlarge variance
                return adjusted_tensor

            # attn_weight = torch.stack([
            #     torch.clip(enhance_tensor(attn_weight, contrast_factor=contrast_strength),
            #                min=0.0, max=1.0)
            #     for head_idx in range(attn_weight.shape[1])
            # ])

            return attn_weight @ V

        def init_attention_processors(unet):
            attn_procs = {}
            number_of_self, number_of_cross = 0, 0
            num_cross_layers = len([name for name in unet.attn_processors.keys() if 'attn2' in name])  # attn1:self-attn attn2:cross-attn
            for i, name in enumerate(unet.attn_processors.keys()):
                is_up_block = 'up' in name
                is_cross_attention = 'attn2' in name
                if is_up_block and is_cross_attention:
                    number_of_cross += 1
                    attn_procs[name] = AttentionProcessor()
                else:
                    attn_procs[name] = DefaultAttentionProcessor()

            unet.set_attn_processor(attn_procs)

        init_attention_processors(unet)


    def x0_vis(self,latent,timestep):
        x0_img = self.latent_to_img_via_vaeDecoder(latent) # PIL Image
        # print(f'{timestep}步,x0_min:{latent.min().item()},x0_mean:{latent.mean().item()},x0_max:{latent.max().item()}')
        savepath = f'./output0_fp16/x0/{timestep}.png'
        x0_img.save(savepath)


    def latent_to_img_via_vaeDecoder(self,latents,return_type='pil'):
        dtype = latents.dtype

        self.vae.to(dtype=torch.float32)

        use_torch_2_0_or_xformers = self.vae.decoder.mid_block.attentions[0].processor in [
            AttnProcessor2_0,
            XFormersAttnProcessor,
            LoRAXFormersAttnProcessor,
            LoRAAttnProcessor2_0,
        ]
        # if xformers or torch_2_0 is used attention block does not need
        # to be in float32 which can save lots of memory
        if not use_torch_2_0_or_xformers:
            self.vae.post_quant_conv.to(dtype)
            self.vae.decoder.conv_in.to(dtype)
            self.vae.decoder.mid_block.to(dtype)
        else:
            latents = latents.float()

        image = self.vae.decode(latents / self.vae.config.scaling_factor, return_dict=False)[0]
        if return_type == 'pt':
            return image # values in [-1.0805,1.0589]
        elif return_type == 'pil':
            # 1:(images / 2 + 0.5).clamp(0, 1)  2:(images * 255).round().astype("uint8")
            image = self.image_processor.postprocess(image, output_type='pil')[0]
        return image

    def img_latent_via_vaeEncoder(self,img):
        latents = self.prepare_pil2latent(img,0,1,1,torch.float16,img.device,add_noise=False)
        return latents

    def adain(self,content, style):
        content_mean = content.mean([2, 3], keepdim=True)
        content_std = content.std([2, 3], keepdim=True) + 1e-5  
        style_mean = style.mean([2, 3], keepdim=True)
        style_std = style.std([2, 3], keepdim=True) + 1e-5

        normalized_content = (content - content_mean) / content_std
        normalized_style = normalized_content * style_std + style_mean
        return normalized_style


    def cut_tensor(self,tensor, patch_size):  
        B, C, height, width = tensor.size()  # tensor [B, C, H, W]
        patch_num_h = height // patch_size  
        patch_num_w = width // patch_size 
        patch_list = []

        # 
        for i in range(patch_num_h):
            for j in range(patch_num_w):
                left = j * patch_size
                upper = i * patch_size
                right = min((j + 1) * patch_size, width)  # 
                lower = min((i + 1) * patch_size, height)  # 
                #
                patch = tensor[:, :, upper:lower, left:right]
                patch_list.append(patch)

        return patch_list

    def patches_to_image(self,patch_list, patch_size):  #
        image_row = image_column = int(len(patch_list)**0.5)

      
        B, C, _, _ = patch_list[0].size()
        image = torch.zeros(B, C, image_row * patch_size, image_column * patch_size)

        for y in range(1, image_row + 1):
            for x in range(1, image_column + 1):
                index = image_column * (y - 1) + x - 1

                if index < len(patch_list):
                    patch = patch_list[index]
                    start_h = (y - 1) * patch_size
                    start_w = (x - 1) * patch_size
                    image[:, :, start_h:start_h + patch_size, start_w:start_w + patch_size] = patch

        return image


    class FreqLoss(torch.nn.Module):
        def __init__(self,new_x0,am_scale,ph_scale, kl_scale, tv_scale, img_latent_via_vaeEncoder):
            super().__init__()
            self.l2loss = torch.nn.MSELoss()
            self.new_x0pred = torch.nn.Parameter(new_x0,requires_grad=True)
            self.am_scale = am_scale
            self.ph_scale = ph_scale
            self.softmax = torch.nn.Softmax()
            self.img_latent_via_vaeEncoder = img_latent_via_vaeEncoder
            self.kl_scale = kl_scale
            self.tv_scale = tv_scale


        def forward(self,old_x0pred,new_x0pred, old_x0pred_latent ):
            old_freq = torch.fft.fft2(old_x0pred,dim=(2,3))
            old_am = torch.abs(old_freq)
            old_ph = torch.angle(old_freq)

            new_freq = torch.fft.fft2(new_x0pred,dim=(2,3))
            new_am = torch.abs(new_freq)
            new_ph = torch.angle(new_freq)

            am_loss = self.l2loss(new_am, old_am)
            ph_loss = self.l2loss(new_ph, old_ph)
            print(f"am_loss.mean: {am_loss.item()}")
            # print(f"ph_loss.mean: {ph_loss.item()}")

            pixel_loss = self.l2loss(old_x0pred, new_x0pred)

            new_x0pred_latent_feat = self.softmax(self.img_latent_via_vaeEncoder(new_x0pred) )
            old_x0pred_latent_feat = self.softmax(old_x0pred_latent)

            if self.kl_scale != 0 :
                kl_div_loss = self.compute_kl_divergence(new_x0pred_latent_feat,old_x0pred_latent_feat)

            if self.tv_scale != 0 :
                tv_loss = self.compute_total_variation_loss(new_x0pred)

            return am_loss * self.am_scale + ph_loss * self.ph_scale + pixel_loss

        def compute_total_variation_loss(self,img):
            tv_h = ((img[:,:,1:,:] - img[:,:,:-1,:]).pow(2)).sum()
            tv_w = ((img[:,:,:,1:] - img[:,:,:,:-1]).pow(2)).sum()
            return (tv_h + tv_w)

        def compute_kl_divergence(self, p, q):
           
            p = p + 1e-10
            q = q + 1e-10

            # use KL_div， reduction='batchmean'
            kl_div = torch.nn.functional.kl_div(p.log(), q, reduction='batchmean')

            return kl_div.item()  

        def get_parameter(self,):
            return self.new_x0pred




    class PhaseLoss(torch.nn.Module):
        def __init__(self,x0, content):
            super().__init__()
            self.x0 = torch.nn.Parameter(x0,requires_grad=True)
            self.content = content
            self.l1loss = torch.nn.L1Loss()

        def forward(self,):
            x0_freq = torch.fft.fft2(self.x0,dim=(2,3))
            x0_ph = torch.angle(x0_freq)

            content_freq = torch.fft.fft2(self.content,dim=(2,3))
            content_ph = torch.angle(content_freq)

            loss = self.l1loss(x0_ph, content_ph)
            # print(f'loss: {loss.item()}')
            return loss

        def get_parameter(self,):
            return self.x0.detach()


