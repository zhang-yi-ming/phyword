import hashlib
import json
import os
import time
import uuid
from typing import Callable, Iterable, Optional, Sequence, Tuple

import torch


PROJECTED_COSMOS_TEXT_SHAPE = (512, 1024)
RAW_COSMOS_TEXT_SHAPE = (512, 100352)
SUPPORTED_COSMOS_TEXT_SHAPES = (
    PROJECTED_COSMOS_TEXT_SHAPE,
    RAW_COSMOS_TEXT_SHAPE,
)
DEFAULT_COSMOS_TEXT_SHAPE = PROJECTED_COSMOS_TEXT_SHAPE
MANIFEST_NAME = "manifest.json"
TENSOR_DIR_NAME = "tensors"


def canonicalize_prompt(prompt: str) -> str:
    return str(prompt).strip()


def prompt_sha256(prompt: str) -> str:
    normalized = canonicalize_prompt(prompt)
    return hashlib.sha256(normalized.encode("utf-8")).hexdigest()


class CosmosTextEmbeddingCache:
    """Sidecar cache for projected or raw Cosmos Qwen text embeddings."""

    def __init__(
        self,
        root: str,
        *,
        expected_shape: Optional[Tuple[int, int]] = None,
        create: bool = False,
    ):
        self.root = os.path.abspath(os.path.expanduser(str(root)))
        if expected_shape is None:
            self.expected_shapes = tuple(SUPPORTED_COSMOS_TEXT_SHAPES)
        else:
            self.expected_shapes = (tuple(int(x) for x in expected_shape),)
        self.expected_shape = self.expected_shapes[0]
        self.tensor_dir = os.path.join(self.root, TENSOR_DIR_NAME)
        self.manifest_path = os.path.join(self.root, MANIFEST_NAME)

        if create:
            os.makedirs(self.tensor_dir, exist_ok=True)
        elif not os.path.isdir(self.tensor_dir):
            raise FileNotFoundError(
                f"Cosmos text cache tensor directory does not exist: {self.tensor_dir}. "
                "Run scripts/precompute_cosmos_text_cache.py first."
            )

    def key(self, prompt: str) -> str:
        return prompt_sha256(prompt)

    def tensor_path(self, prompt: str) -> str:
        return self.tensor_path_for_key(self.key(prompt))

    def tensor_path_for_key(self, key: str) -> str:
        return os.path.join(self.tensor_dir, f"{key}.pt")

    def contains(self, prompt: str) -> bool:
        return os.path.exists(self.tensor_path(prompt))

    def load(self, prompt: str, *, map_location: str | torch.device = "cpu") -> torch.Tensor:
        key = self.key(prompt)
        path = self.tensor_path_for_key(key)
        if not os.path.exists(path):
            raise FileNotFoundError(
                "Missing cached Cosmos text embedding for "
                f"sha256={key}, prompt={canonicalize_prompt(prompt)!r}. "
                "Run scripts/precompute_cosmos_text_cache.py with this dataset/cache path."
            )

        tensor = torch.load(path, map_location=map_location)
        if isinstance(tensor, dict) and "embedding" in tensor:
            tensor = tensor["embedding"]
        return self._validate_tensor(tensor, prompt=prompt)

    def save(
        self,
        prompt: str,
        tensor: torch.Tensor,
        *,
        overwrite: bool = False,
        metadata: Optional[dict] = None,
    ) -> str:
        saved = self.save_many(
            [(prompt, tensor)],
            overwrite=overwrite,
            metadata=metadata,
        )
        return saved[0] if saved else self.tensor_path(prompt)

    def save_many(
        self,
        prompt_tensor_pairs: Iterable[Tuple[str, torch.Tensor]],
        *,
        overwrite: bool = False,
        metadata: Optional[dict] = None,
    ) -> list[str]:
        os.makedirs(self.tensor_dir, exist_ok=True)
        manifest = self._load_manifest()
        entries = manifest.setdefault("entries", {})
        saved_paths = []

        for prompt, tensor in prompt_tensor_pairs:
            normalized = canonicalize_prompt(prompt)
            key = self.key(normalized)
            path = self.tensor_path_for_key(key)
            if os.path.exists(path) and not overwrite:
                continue

            tensor = self._validate_tensor(tensor, prompt=normalized).detach().cpu().to(torch.bfloat16)
            tmp_path = f"{path}.tmp.{os.getpid()}.{uuid.uuid4().hex}"
            torch.save(tensor, tmp_path)
            os.replace(tmp_path, path)

            entry = {
                "prompt": normalized,
                "tensor": os.path.relpath(path, self.root),
                "shape": list(tensor.shape),
                "dtype": str(tensor.dtype),
                "embedding_format": cosmos_text_embedding_format(tensor),
                "updated_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
            }
            if metadata:
                entry.update(metadata)
            entries[key] = entry
            saved_paths.append(path)

        manifest["num_entries"] = len(entries)
        manifest["expected_shape"] = list(self.expected_shape)
        manifest["expected_shapes"] = [list(shape) for shape in self.expected_shapes]
        manifest["format_version"] = 1
        manifest["tensor_dir"] = TENSOR_DIR_NAME
        self._save_manifest(manifest)
        return saved_paths

    def get_or_compute(
        self,
        prompt: str,
        compute_fn: Callable[[str], torch.Tensor],
        *,
        map_location: str | torch.device = "cpu",
    ) -> torch.Tensor:
        if self.contains(prompt):
            return self.load(prompt, map_location=map_location)

        normalized = canonicalize_prompt(prompt)
        tensor = compute_fn(normalized)
        self.save(normalized, tensor, overwrite=True)
        return self.load(normalized, map_location=map_location)

    def _validate_tensor(self, tensor: torch.Tensor, *, prompt: str = "") -> torch.Tensor:
        if not torch.is_tensor(tensor):
            raise TypeError(f"Cached Cosmos text embedding must be a tensor, got {type(tensor)!r}.")
        shape = tuple(int(x) for x in tensor.shape)
        if shape not in self.expected_shapes:
            prompt_note = f" for prompt={canonicalize_prompt(prompt)!r}" if prompt else ""
            raise ValueError(
                f"Expected Cosmos text embedding shape in {self.expected_shapes}{prompt_note}, "
                f"got {shape}."
            )
        return tensor

    def _load_manifest(self) -> dict:
        if not os.path.exists(self.manifest_path):
            return {}
        with open(self.manifest_path, "r") as f:
            return json.load(f)

    def _save_manifest(self, manifest: dict) -> None:
        os.makedirs(self.root, exist_ok=True)
        tmp_path = f"{self.manifest_path}.tmp.{os.getpid()}.{uuid.uuid4().hex}"
        with open(tmp_path, "w") as f:
            json.dump(manifest, f, indent=2, ensure_ascii=False)
        os.replace(tmp_path, self.manifest_path)


def cosmos_text_embedding_format(tensor: torch.Tensor) -> str:
    shape = tuple(int(x) for x in tensor.shape)
    if shape == RAW_COSMOS_TEXT_SHAPE:
        return "raw_qwen_full_concat"
    if shape == PROJECTED_COSMOS_TEXT_SHAPE:
        return "projected_crossattn"
    return "unknown"


class CosmosQwenTextEmbedder:
    """Compute raw native Reason1/Qwen Cosmos text embeddings."""

    def __init__(
        self,
        cosmos_dit: torch.nn.Module,
        text_encoder_config,
        *,
        device: str | torch.device = "cuda",
        output_dtype: torch.dtype = torch.bfloat16,
    ):
        self.cosmos_dit = cosmos_dit
        self.text_encoder_config = text_encoder_config
        self.device = torch.device(device)
        self.output_dtype = output_dtype
        self._text_encoder = None

        if self.device.type != "cuda" or not torch.cuda.is_available():
            raise RuntimeError(
                "The native Cosmos Reason1/Qwen text encoder currently requires CUDA "
                "because its tokenizer path moves input_ids to device='cuda'."
            )

    def compute_one(self, prompt: str) -> torch.Tensor:
        return self.compute_batch([prompt])[0]

    def compute_batch(self, prompts: Sequence[str]) -> torch.Tensor:
        if not prompts:
            raise ValueError("prompts must not be empty.")
        if self.device.type == "cuda":
            torch.cuda.set_device(self.device)

        text_encoder = self._ensure_text_encoder()
        with torch.no_grad():
            raw_embeddings = text_encoder.compute_text_embeddings_online(
                {"input_prompt": [canonicalize_prompt(prompt) for prompt in prompts]},
                "input_prompt",
            )
        return raw_embeddings.detach().cpu().to(self.output_dtype)

    def _ensure_text_encoder(self):
        if self._text_encoder is None:
            from cosmos_predict2._src.predict2.text_encoders.text_encoder import TextEncoder

            self._text_encoder = TextEncoder(self.text_encoder_config, device=str(self.device))
        return self._text_encoder
