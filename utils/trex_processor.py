"""Utilities for loading the T-Rex/Qwen-VL processor reliably."""

import json
import os
from typing import Any, Iterable, List

import torch
from transformers import AutoImageProcessor, AutoProcessor, AutoTokenizer
from transformers.feature_extraction_utils import BatchFeature


class TrexQwenVLProcessor:
    """Fallback processor for T-Rex checkpoints saved as loose components.

    Some checkpoints contain a Qwen tokenizer and image preprocessor, but the
    installed Transformers version cannot rehydrate them as a full Qwen-VL
    processor. This wrapper exposes the small surface used by training/eval:
    tokenizer, image_processor, apply_chat_template, __call__, and
    save_pretrained.
    """

    image_token = "<|image_pad|>"

    def __init__(self, tokenizer: Any, image_processor: Any, processor_class: str = "Qwen3VLProcessor"):
        self.tokenizer = tokenizer
        self.image_processor = image_processor
        self.processor_class = processor_class
        self.pad_id = getattr(tokenizer, "pad_token_id", None)
        if self.pad_id is None and getattr(tokenizer, "eos_token_id", None) is not None:
            tokenizer.pad_token = tokenizer.eos_token
            self.pad_id = tokenizer.pad_token_id

    def __getattr__(self, name: str) -> Any:
        return getattr(self.tokenizer, name)

    def apply_chat_template(self, *args: Any, **kwargs: Any) -> Any:
        return self.tokenizer.apply_chat_template(*args, **kwargs)

    def save_pretrained(self, save_directory: str, **kwargs: Any) -> None:
        os.makedirs(save_directory, exist_ok=True)
        self.tokenizer.save_pretrained(save_directory, **kwargs)
        self.image_processor.save_pretrained(save_directory, **kwargs)
        processor_config_path = os.path.join(save_directory, "processor_config.json")
        if not os.path.exists(processor_config_path):
            with open(processor_config_path, "w", encoding="utf-8") as f:
                json.dump({"processor_class": self.processor_class}, f, indent=2)

    def _as_text_list(self, text: Any) -> tuple[List[str], bool]:
        if text is None:
            return [], False
        if isinstance(text, str):
            return [text], True
        return list(text), False

    def _image_token_repeats(self, image_grid_thw: Any) -> List[int]:
        if image_grid_thw is None:
            return []
        if torch.is_tensor(image_grid_thw):
            grids: Iterable[Any] = image_grid_thw.detach().cpu().tolist()
        else:
            grids = image_grid_thw

        merge_size = int(getattr(self.image_processor, "merge_size", 2) or 1)
        merge_area = max(merge_size * merge_size, 1)
        repeats: List[int] = []
        for grid in grids:
            prod = 1
            for value in grid:
                prod *= int(value)
            repeats.append(max(prod // merge_area, 1))
        return repeats

    def _expand_image_tokens(self, texts: List[str], image_grid_thw: Any) -> List[str]:
        repeats = self._image_token_repeats(image_grid_thw)
        if not repeats:
            return texts

        expanded = []
        image_index = 0
        placeholder = "<|trex_image_placeholder|>"
        for text in texts:
            while self.image_token in text:
                if image_index >= len(repeats):
                    break
                text = text.replace(self.image_token, placeholder * repeats[image_index], 1)
                image_index += 1
            expanded.append(text.replace(placeholder, self.image_token))
        return expanded

    def __call__(self, text: Any = None, images: Any = None, return_tensors: str | None = None, **kwargs: Any):
        padding = kwargs.pop("padding", False)
        tokenizer_keys = {
            "add_special_tokens",
            "truncation",
            "max_length",
            "pad_to_multiple_of",
            "return_attention_mask",
        }
        tokenizer_kwargs = {
            key: kwargs.pop(key)
            for key in list(kwargs.keys())
            if key in tokenizer_keys
        }

        image_inputs = {}
        if images is not None:
            image_inputs = self.image_processor(images=images, return_tensors=return_tensors)

        texts, was_single = self._as_text_list(text)
        if texts:
            texts = self._expand_image_tokens(texts, image_inputs.get("image_grid_thw"))
            tokenizer_input = texts[0] if was_single else texts
            text_inputs = self.tokenizer(
                tokenizer_input,
                return_tensors=return_tensors,
                padding=padding,
                **tokenizer_kwargs,
            )
        else:
            text_inputs = {}

        data = {}
        data.update(text_inputs)
        data.update(image_inputs)
        return BatchFeature(data=data, tensor_type=return_tensors)


def _read_processor_class(processor_path: str) -> str:
    for filename in ("processor_config.json", "preprocessor_config.json", "tokenizer_config.json"):
        path = os.path.join(processor_path, filename)
        if not os.path.exists(path):
            continue
        try:
            with open(path, "r", encoding="utf-8") as f:
                config = json.load(f)
        except Exception:
            continue
        processor_class = config.get("processor_class")
        if processor_class:
            return str(processor_class)
    return "Qwen3VLProcessor"


def _try_build_official_processor(tokenizer: Any, image_processor: Any, processor_class: str, processor_path: str):
    import transformers

    video_processor = None
    auto_video_processor = getattr(transformers, "AutoVideoProcessor", None)
    if auto_video_processor is not None:
        try:
            video_processor = auto_video_processor.from_pretrained(processor_path, trust_remote_code=True)
        except Exception:
            video_processor = None

    candidates = [processor_class, "Qwen3VLProcessor", "Qwen2VLProcessor", "Qwen2_5_VLProcessor"]
    for class_name in dict.fromkeys(candidates):
        processor_cls = getattr(transformers, class_name, None)
        if processor_cls is None:
            continue

        kwargs = {"image_processor": image_processor, "tokenizer": tokenizer}
        chat_template = getattr(tokenizer, "chat_template", None)
        if chat_template is not None:
            kwargs["chat_template"] = chat_template
        if video_processor is not None:
            kwargs["video_processor"] = video_processor
        try:
            return processor_cls(**kwargs)
        except Exception:
            continue
    return None


def load_trex_processor(processor_path: str, trust_remote_code: bool = True):
    """Load a full T-Rex/Qwen-VL processor while preserving the T-Rex tokenizer."""

    processor = AutoProcessor.from_pretrained(processor_path, trust_remote_code=trust_remote_code)
    if hasattr(processor, "tokenizer") and hasattr(processor, "image_processor"):
        return processor

    if hasattr(processor, "convert_tokens_to_ids"):
        tokenizer = processor
    else:
        tokenizer = AutoTokenizer.from_pretrained(processor_path, trust_remote_code=trust_remote_code)

    image_processor = getattr(processor, "image_processor", None)
    if image_processor is None:
        image_processor = AutoImageProcessor.from_pretrained(
            processor_path,
            trust_remote_code=trust_remote_code,
        )

    processor_class = _read_processor_class(processor_path)
    official_processor = _try_build_official_processor(
        tokenizer=tokenizer,
        image_processor=image_processor,
        processor_class=processor_class,
        processor_path=processor_path,
    )
    if official_processor is not None:
        return official_processor

    return TrexQwenVLProcessor(
        tokenizer=tokenizer,
        image_processor=image_processor,
        processor_class=processor_class,
    )
