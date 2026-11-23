from pathlib import Path
from typing import Any

import cv2
import numpy as np
import torch
from diffusers import StableDiffusionInstructPix2PixPipeline
from peft import PeftModel
from PIL import Image

import config


def load_pipeline() -> StableDiffusionInstructPix2PixPipeline:
    pipe = StableDiffusionInstructPix2PixPipeline.from_pretrained(
        config.MODEL_ID,
        torch_dtype=torch.float16,
    )
    pipe.unet = PeftModel.from_pretrained(pipe.unet, config.OUTPUT_DIR)
    return pipe.to("cuda" if torch.cuda.is_available() else "cpu")


def compute_mas(
    orig_pil: Image.Image,
    edited_pil: Image.Image,
    gt_pil: Image.Image,
) -> float:
    def to_grayscale(image: Image.Image) -> np.ndarray:
        image = image.convert("RGB").resize((256, 256), Image.Resampling.BICUBIC)
        rgb = np.asarray(image)
        return cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)

    original = to_grayscale(orig_pil)
    edited = to_grayscale(edited_pil)
    ground_truth = to_grayscale(gt_pil)

    flow_params: dict[str, Any] = {
        "pyr_scale": 0.5,
        "levels": 3,
        "winsize": 15,
        "iterations": 3,
        "poly_n": 5,
        "poly_sigma": 1.2,
        "flags": 0,
    }
    edited_flow = cv2.calcOpticalFlowFarneback(
        original, edited, None, **flow_params
    )
    ground_truth_flow = cv2.calcOpticalFlowFarneback(
        original, ground_truth, None, **flow_params
    )

    edited_magnitude = np.linalg.norm(edited_flow, axis=2)
    ground_truth_magnitude = np.linalg.norm(ground_truth_flow, axis=2)
    max_displacement = np.hypot(256, 256)
    magnitude_score = 1.0 - np.mean(
        np.abs(edited_magnitude - ground_truth_magnitude) / max_displacement
    )
    magnitude_score = float(np.clip(magnitude_score, 0.0, 1.0))

    edited_nonzero = edited_magnitude > 1e-6
    ground_truth_nonzero = ground_truth_magnitude > 1e-6
    jointly_moving = edited_nonzero & ground_truth_nonzero
    if np.any(jointly_moving):
        dot_product = np.sum(edited_flow * ground_truth_flow, axis=2)
        denominator = edited_magnitude * ground_truth_magnitude
        cosine_similarity = dot_product[jointly_moving] / denominator[jointly_moving]
        direction_score = float(
            np.mean((np.clip(cosine_similarity, -1.0, 1.0) + 1.0) / 2.0)
        )
    else:
        direction_score = float(np.all(~edited_nonzero & ~ground_truth_nonzero))

    return 100.0 * (magnitude_score + direction_score) / 2.0


def edit_image(
    pipe: StableDiffusionInstructPix2PixPipeline,
    image_path: str | Path,
    prompt: str,
) -> Image.Image:
    image_path = Path(image_path)
    with Image.open(image_path) as source:
        image = source.convert("RGB").resize(
            (512, 512), Image.Resampling.LANCZOS
        )

    edited_image = pipe(prompt=prompt, image=image).images[0]
    output_path = Path(config.OUTPUT_DIR) / f"edited_{image_path.stem}.png"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    edited_image.save(output_path)
    return edited_image
