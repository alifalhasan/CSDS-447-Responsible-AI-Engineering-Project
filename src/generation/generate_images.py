#!/usr/bin/env python3
"""
Image Generation Module for Negative Role Fairness Analysis

This module handles text-to-image generation using various models including
Stable Diffusion, SDXL, FLUX, and DALL-E for bias analysis in negative role depictions.
"""

import os
import json
import argparse
import logging
from typing import List, Dict, Any, Optional
from pathlib import Path
import pandas as pd
from PIL import Image
import torch
from diffusers import (
    StableDiffusionPipeline,
    StableDiffusionXLPipeline,
    StableDiffusionXLImg2ImgPipeline,
    FluxPipeline,
    DPMSolverMultistepScheduler,
    EulerDiscreteScheduler,
)
import requests
from io import BytesIO

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ImageGenerator:
    """Main class for generating images from text prompts."""

    def __init__(
        self,
        model_name: str = "stable-diffusion-v1-5",
        device: str = "auto",
        use_refiner: bool = False,
    ):
        """
        Initialize the image generator.

        Args:
            model_name: Name of the model to use
            device: Device to run on ('auto', 'cpu', 'cuda', 'mps')
            use_refiner: Whether to use SDXL refiner (only for SDXL models)
        """
        self.model_name = model_name
        self.device = self._get_device(device)
        self.use_refiner = use_refiner
        self.pipeline = None
        self.refiner = None
        self.is_sdxl = "xl" in model_name.lower()
        self.is_flux = "flux" in model_name.lower()
        self._load_model()

    def _get_device(self, device: str) -> str:
        """Determine the best device to use."""
        if device == "auto":
            if torch.cuda.is_available():
                return "cuda"
            elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
                return "mps"
            else:
                return "cpu"
        return device

    def _load_model(self):
        """Load the specified model."""
        try:
            if self.is_flux:
                self._load_flux()
            elif "stable-diffusion" in self.model_name.lower():
                self._load_stable_diffusion()
            elif "dall-e" in self.model_name.lower():
                self._load_dalle()
            else:
                raise ValueError(f"Unsupported model: {self.model_name}")
            logger.info(f"Successfully loaded {self.model_name} on {self.device}")
        except Exception as e:
            logger.error(f"Failed to load model {self.model_name}: {e}")
            raise

    def _load_flux(self):
        """Load FLUX.1 model."""
        logger.info(f"Loading FLUX model: {self.model_name}")

        # Determine appropriate dtype
        if self.device == "cuda":
            dtype = torch.bfloat16  # FLUX works best with bfloat16 on CUDA
        else:
            dtype = torch.float32

        # Load FLUX pipeline
        self.pipeline = FluxPipeline.from_pretrained(self.model_name, torch_dtype=dtype)

        self.pipeline = self.pipeline.to(self.device)

        # Enable memory optimizations
        if hasattr(self.pipeline, "enable_attention_slicing"):
            self.pipeline.enable_attention_slicing()

        if hasattr(self.pipeline, "enable_vae_slicing"):
            self.pipeline.enable_vae_slicing()

        logger.info(f"Successfully loaded FLUX model on {self.device}")

    def _load_stable_diffusion(self):
        """Load Stable Diffusion model."""
        if self.is_sdxl:
            self._load_sdxl()
        else:
            self._load_sd_base()

    def _load_sd_base(self):
        """Load base Stable Diffusion model (v1.5 or v2.1)."""
        model_id = "runwayml/stable-diffusion-v1-5"
        if "v2" in self.model_name.lower():
            model_id = "stabilityai/stable-diffusion-2-1"

        logger.info(f"Loading Stable Diffusion model: {model_id}")

        self.pipeline = StableDiffusionPipeline.from_pretrained(
            model_id,
            torch_dtype=torch.float16 if self.device == "cuda" else torch.float32,
            safety_checker=None,  # Disable safety checker for research purposes
            requires_safety_checker=False,
        )

        # Use DPMSolver for faster, higher quality generation
        self.pipeline.scheduler = DPMSolverMultistepScheduler.from_config(
            self.pipeline.scheduler.config
        )

        self.pipeline = self.pipeline.to(self.device)

        # Enable memory efficient attention if available
        if hasattr(self.pipeline, "enable_memory_efficient_attention"):
            self.pipeline.enable_memory_efficient_attention()

        logger.info(f"Successfully loaded {model_id} on {self.device}")

    def _load_sdxl(self):
        """Load Stable Diffusion XL model with optional refiner."""
        base_model_id = "stabilityai/stable-diffusion-xl-base-1.0"
        refiner_model_id = "stabilityai/stable-diffusion-xl-refiner-1.0"

        logger.info(f"Loading Stable Diffusion XL base model: {base_model_id}")

        # Determine appropriate dtype
        if self.device == "cuda":
            dtype = torch.float16
        elif self.device == "mps":
            dtype = torch.float32  # MPS doesn't support float16 well
        else:
            dtype = torch.float32

        # Load base SDXL pipeline
        self.pipeline = StableDiffusionXLPipeline.from_pretrained(
            base_model_id,
            torch_dtype=dtype,
            use_safetensors=True,
            variant="fp16" if dtype == torch.float16 else None,
        )

        # Use Euler scheduler for SDXL (often better quality)
        self.pipeline.scheduler = EulerDiscreteScheduler.from_config(
            self.pipeline.scheduler.config
        )

        self.pipeline = self.pipeline.to(self.device)

        # Enable memory efficient attention if available
        if hasattr(self.pipeline, "enable_attention_slicing"):
            self.pipeline.enable_attention_slicing()

        if hasattr(self.pipeline, "enable_model_cpu_offload") and self.device == "cpu":
            # For CPU, enable model offloading to save memory
            try:
                self.pipeline.enable_model_cpu_offload()
            except:
                pass

        logger.info(f"Successfully loaded SDXL base model on {self.device}")

        # Load refiner if requested
        if self.use_refiner:
            logger.info(f"Loading SDXL refiner model: {refiner_model_id}")
            try:
                self.refiner = StableDiffusionXLImg2ImgPipeline.from_pretrained(
                    refiner_model_id,
                    torch_dtype=dtype,
                    use_safetensors=True,
                    variant="fp16" if dtype == torch.float16 else None,
                )
                self.refiner = self.refiner.to(self.device)

                if hasattr(self.refiner, "enable_attention_slicing"):
                    self.refiner.enable_attention_slicing()

                logger.info(f"Successfully loaded SDXL refiner on {self.device}")
            except Exception as e:
                logger.warning(
                    f"Failed to load refiner: {e}. Continuing without refiner."
                )
                self.use_refiner = False

    def _load_dalle(self):
        """Load DALL-E model (placeholder for API integration)."""
        # This would integrate with OpenAI's DALL-E API
        # For now, we'll use a placeholder
        logger.warning(
            "DALL-E integration not implemented yet. Using Stable Diffusion instead."
        )
        self._load_stable_diffusion()

    def generate_image(
        self,
        prompt: str,
        num_images: int = 1,
        guidance_scale: float = 7.5,
        num_inference_steps: int = 20,
        seed: int = None,
        negative_prompt: Optional[str] = None,
        height: Optional[int] = None,
        width: Optional[int] = None,
    ) -> List[Image.Image]:
        """
        Generate images from a text prompt.

        Args:
            prompt: Text prompt for image generation
            num_images: Number of images to generate
            guidance_scale: How closely to follow the prompt
            num_inference_steps: Number of denoising steps
            seed: Random seed for reproducibility
            negative_prompt: Negative prompt to avoid certain features
            height: Image height (default: 1024 for SDXL/FLUX, 512 for SD)
            width: Image width (default: 1024 for SDXL/FLUX, 512 for SD)

        Returns:
            List of generated PIL Images
        """
        if self.pipeline is None:
            raise RuntimeError("Model not loaded")

        # Set seed for reproducibility
        if seed is not None:
            generator = torch.Generator(device=self.device)
            generator.manual_seed(seed)
        else:
            generator = None

        # Set default dimensions based on model
        if height is None:
            height = 1024 if (self.is_sdxl or self.is_flux) else 512
        if width is None:
            width = 1024 if (self.is_sdxl or self.is_flux) else 512

        try:
            if self.is_flux:
                # FLUX generation
                with torch.autocast(
                    self.device if self.device != "mps" else "cpu",
                    dtype=torch.bfloat16 if self.device == "cuda" else torch.float32,
                ):
                    images = self.pipeline(
                        prompt=prompt,
                        num_images_per_prompt=num_images,
                        guidance_scale=guidance_scale,
                        num_inference_steps=num_inference_steps,
                        height=height,
                        width=width,
                        generator=generator,
                    ).images

            elif self.is_sdxl:
                # Set default negative prompt for SDXL if not provided
                if negative_prompt is None:
                    negative_prompt = "blurry, low quality, distorted, deformed, ugly, bad anatomy, bad proportions"

                # Adjust guidance scale for SDXL (typically uses 5-7)
                if guidance_scale == 7.5:
                    guidance_scale = 5.0

                # SDXL generation
                with torch.autocast(
                    self.device if self.device != "mps" else "cpu",
                    dtype=torch.float16 if self.device == "cuda" else torch.float32,
                ):
                    images = self.pipeline(
                        prompt=prompt,
                        negative_prompt=negative_prompt,
                        num_images_per_prompt=num_images,
                        guidance_scale=guidance_scale,
                        num_inference_steps=num_inference_steps,
                        height=height,
                        width=width,
                        generator=generator,
                    ).images

                # Apply refiner if available
                if self.use_refiner and self.refiner is not None:
                    logger.info("Applying SDXL refiner for enhanced quality...")
                    refined_images = []
                    for image in images:
                        refined = self.refiner(
                            prompt=prompt,
                            negative_prompt=negative_prompt,
                            image=image,
                            num_inference_steps=num_inference_steps // 2,
                            strength=0.3,
                            guidance_scale=guidance_scale,
                            generator=generator,
                        ).images[0]
                        refined_images.append(refined)
                    images = refined_images
            else:
                # Standard SD generation
                with torch.autocast(
                    self.device if self.device != "mps" else "cpu",
                    dtype=torch.float16 if self.device == "cuda" else torch.float32,
                ):
                    images = self.pipeline(
                        prompt=prompt,
                        negative_prompt=negative_prompt,
                        num_images_per_prompt=num_images,
                        guidance_scale=guidance_scale,
                        num_inference_steps=num_inference_steps,
                        height=height,
                        width=width,
                        generator=generator,
                    ).images

            return images
        except Exception as e:
            logger.error(f"Error generating image for prompt '{prompt}': {e}")
            import traceback

            logger.error(traceback.format_exc())
            return []

    def generate_batch(
        self,
        prompts: List[str],
        output_dir: str,
        num_images_per_prompt: int = 5,
        **kwargs,
    ) -> Dict[str, List[str]]:
        """
        Generate images for a batch of prompts.

        Args:
            prompts: List of text prompts
            output_dir: Directory to save images
            num_images_per_prompt: Number of images per prompt
            **kwargs: Additional arguments for generate_image

        Returns:
            Dictionary mapping prompts to list of saved image paths
        """
        os.makedirs(output_dir, exist_ok=True)
        results = {}

        for i, prompt in enumerate(prompts):
            logger.info(
                f"Generating images for prompt {i+1}/{len(prompts)}: {prompt[:50]}..."
            )

            # Clean prompt for filename
            safe_prompt = "".join(
                c for c in prompt if c.isalnum() or c in (" ", "-", "_")
            ).rstrip()
            safe_prompt = safe_prompt[:50]  # Limit filename length

            images = self.generate_image(prompt, num_images_per_prompt, **kwargs)
            saved_paths = []

            for j, image in enumerate(images):
                filename = f"{i:03d}_{safe_prompt}_{j:02d}.png"
                filepath = os.path.join(output_dir, filename)
                image.save(filepath)
                saved_paths.append(filepath)

            results[prompt] = saved_paths
            logger.info(f"Saved {len(images)} images for prompt {i+1}")

        return results


def load_prompts_from_csv(csv_path: str) -> List[Dict[str, Any]]:
    """Load prompts from CSV file."""
    df = pd.read_csv(csv_path)
    df.columns = df.columns.str.strip()
    prompts = []

    for _, row in df.iterrows():
        prompt_data = {
            "prompt_id": row["prompt_id"],
            "category": row["category"],
            "role_label": row["role_label"],
            "gender_spec": row["gender_spec"],
            "style": row["style"],
            "context": row["context"],
            "natural_prompt": row["natural_prompt"],
            "token_prompt": row["token_prompt"],
            "mitigation": row["mitigation"],
            "notes": row["notes"],
        }
        prompts.append(prompt_data)

    return prompts


def main():
    """Main function for command-line usage."""
    parser = argparse.ArgumentParser(description="Generate images for bias analysis")
    parser.add_argument(
        "--model", default="stable-diffusion-v1-5", help="Model to use for generation"
    )
    parser.add_argument(
        "--prompts", required=True, help="Path to CSV file containing prompts"
    )
    parser.add_argument(
        "--output", required=True, help="Output directory for generated images"
    )
    parser.add_argument(
        "--num-images", type=int, default=5, help="Number of images per prompt"
    )
    parser.add_argument(
        "--device", default="auto", help="Device to use (auto, cpu, cuda)"
    )
    parser.add_argument(
        "--seed", type=int, default=42, help="Random seed for reproducibility"
    )
    parser.add_argument(
        "--use-refiner",
        action="store_true",
        help="Use SDXL refiner for enhanced quality (SDXL only)",
    )

    args = parser.parse_args()

    # Load prompts
    prompts_data = load_prompts_from_csv(args.prompts)
    prompts = [p["natural_prompt"] for p in prompts_data]

    # Initialize generator
    generator = ImageGenerator(
        model_name=args.model, device=args.device, use_refiner=args.use_refiner
    )

    # Generate images
    results = generator.generate_batch(
        prompts=prompts,
        output_dir=args.output,
        num_images_per_prompt=args.num_images,
        seed=args.seed,
    )

    # Save results metadata
    metadata = {
        "model": args.model,
        "device": generator.device,
        "num_prompts": len(prompts),
        "num_images_per_prompt": args.num_images,
        "seed": args.seed,
        "results": results,
    }

    metadata_path = os.path.join(args.output, "generation_metadata.json")
    with open(metadata_path, "w") as f:
        json.dump(metadata, f, indent=2)

    logger.info(f"Generation complete! Results saved to {args.output}")
    logger.info(
        f"Generated {sum(len(paths) for paths in results.values())} total images"
    )


if __name__ == "__main__":
    main()
