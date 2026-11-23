from itertools import islice
from pathlib import Path
from typing import Any, Dict, Iterable, Mapping, Sequence

import torch
import numpy as np
from datasets import load_dataset
from PIL import Image
from torch.utils.data import DataLoader, Dataset

import config


def _is_image_value(value: Any) -> bool:
    if isinstance(value, Image.Image):
        return True
    if isinstance(value, (bytes, bytearray)):
        return True
    if isinstance(value, dict):
        return any(key in value for key in ("bytes", "path", "data"))
    if isinstance(value, (list, tuple)):
        return bool(value) and isinstance(value[0], (bytes, bytearray, Image.Image))
    return hasattr(value, "mode") and hasattr(value, "size")


def _detect_keys(sample: Mapping[str, Any]) -> Dict[str, str]:
    items = list(sample.items())

    image_keys = [key for key, value in items if _is_image_value(value)]
    prompt_names = (
        "prompt", "instruction", "caption", "text", "description",
        "edited_prompt", "edit_prompt", "motion",
    )

    prompt_key = next(
        (key for key in sample if str(key).lower() in prompt_names),
        None,
    )
    if prompt_key is None:
        prompt_key = next(
            (
                key for key, value in items
                if isinstance(value, str)
                and str(key).lower() not in {"image", "source", "target"}
            ),
            None,
        )
    if prompt_key is None:
        raise ValueError("Could not detect a prompt/text column in the first dataset sample")

    def ranked_key(role: str, candidates: Iterable[str]) -> str:
        role_names = [name.lower() for name in candidates]
        for key in image_keys:
            name = str(key).lower()
            if role in name or any(token in name for token in role_names):
                return key
        return ""

    source_key = ranked_key(
        "source", ("source", "src", "input", "before", "original", "start")
    )
    target_key = ranked_key(
        "target", ("target", "tgt", "output", "after", "edited", "result", "end")
    )

    if not source_key or not target_key:
        remaining = [key for key in image_keys if key not in (source_key, target_key)]
        if not source_key and remaining:
            source_key = remaining[0]
        if not target_key:
            for key in image_keys:
                if key != source_key:
                    target_key = key
                    break

    if not source_key or not target_key or source_key == target_key:
        raise ValueError(
            f"Could not detect source and target image columns; found image columns: {image_keys}"
        )

    return {
        "source_image": source_key,
        "target_image": target_key,
        "prompt": prompt_key,
    }


def _to_pil(value: Any) -> Image.Image:
    if isinstance(value, dict):
        if value.get("bytes") is not None:
            from io import BytesIO

            value = BytesIO(value["bytes"])
        elif value.get("path") is not None:
            value = value["path"]
        elif value.get("data") is not None:
            value = value["data"]

    if isinstance(value, (bytes, bytearray)):
        from io import BytesIO

        value = BytesIO(value)
    if isinstance(value, str) and Path(value).is_file():
        value = Image.open(value)
    if not isinstance(value, Image.Image):
        raise TypeError(f"Unsupported image value: {type(value).__name__}")
    return value.convert("RGB")


class MotionEditDataset(Dataset):
    def __init__(
        self,
        samples: Sequence[Mapping[str, Any]],
        keys_dict: Mapping[str, str],
        resolution: int = config.RESOLUTION,
    ) -> None:
        self.samples = samples
        self.keys_dict = dict(keys_dict)
        self.resolution = resolution
        self.source_key = self.keys_dict["source_image"]
        self.target_key = self.keys_dict["target_image"]
        self.prompt_key = self.keys_dict["prompt"]

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> Dict[str, Any]:
        sample = self.samples[index]
        source = _to_pil(sample[self.source_key])
        target = _to_pil(sample[self.target_key])
        size = (self.resolution, self.resolution)

        source = source.resize(size, Image.Resampling.BICUBIC)
        target = target.resize(size, Image.Resampling.BICUBIC)

        source_tensor = torch.from_numpy(
            np.array(source, dtype="float32", copy=True)
        ).permute(2, 0, 1).div_(255.0).mul_(2.0).sub_(1.0)
        target_tensor = torch.from_numpy(
            np.array(target, dtype="float32", copy=True)
        ).permute(2, 0, 1).div_(255.0).mul_(2.0).sub_(1.0)

        return {
            "source_image": source_tensor,
            "target_image": target_tensor,
            "prompt": str(sample[self.prompt_key]),
        }


def get_data():
    total = config.NUM_TRAIN + config.NUM_TEST
    stream = load_dataset(
        config.DATASET_NAME,
        split="train",
        streaming=True,
    )
    samples = list(islice(stream, total))
    if len(samples) != total:
        raise ValueError(
            f"Dataset '{config.DATASET_NAME}' returned {len(samples)} of {total} requested samples"
        )

    keys_dict = _detect_keys(samples[0])
    train_samples = samples[: config.NUM_TRAIN]
    test_samples = samples[config.NUM_TRAIN :]
    train_dataset = MotionEditDataset(train_samples, keys_dict)

    train_dataloader = DataLoader(
        train_dataset,
        batch_size=4,
        shuffle=True,
        num_workers=0,
        pin_memory=torch.cuda.is_available(),
    )
    return train_dataloader, test_samples, keys_dict
