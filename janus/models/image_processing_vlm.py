# Copyright (c) 2023-2024 DeepSeek.
#
# Permission is hereby granted, free of charge, to any person obtaining a copy of
# this software and associated documentation files (the "Software"), to deal in
# the Software without restriction, including without limitation the rights to
# use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of
# the Software, and to permit persons to whom the Software is furnished to do so,
# subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS
# FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR
# COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER
# IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN
# CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

from typing import List, Tuple, Union
import copy

import numpy as np
import torch
import torchvision
import torchvision.transforms.functional
from PIL import Image
from transformers import AutoImageProcessor, PretrainedConfig
from transformers.image_processing_utils import BaseImageProcessor, BatchFeature
from transformers.utils import logging

logger = logging.get_logger(__name__)

ImageType = Union[np.ndarray, torch.Tensor, Image.Image]
IMAGENET_MEAN = (0.48145466, 0.4578275, 0.40821073)
IMAGENET_STD = (0.26862954, 0.26130258, 0.27577711)
IMAGENET_INCEPTION_MEAN = (0.5, 0.5, 0.5)
IMAGENET_INCEPTION_STD = (0.5, 0.5, 0.5)


def expand2square(pil_img, background_color):
    width, height = pil_img.size
    if width == height:
        return pil_img
    elif width > height:
        result = Image.new(pil_img.mode, (width, width), background_color)
        result.paste(pil_img, (0, (width - height) // 2))
        return result
    else:
        result = Image.new(pil_img.mode, (height, height), background_color)
        result.paste(pil_img, ((height - width) // 2, 0))
        return result


class VLMImageProcessorConfig(PretrainedConfig):
    model_type = "deepseek_vlm"
    image_size: int
    min_size: int
    image_mean: Union[Tuple[float, float, float], List[float]]
    image_std: Union[Tuple[float, float, float], List[float]]
    rescale_factor: float
    do_normalize: bool

    def __init__(
        self,
        image_size: int,
        min_size: int = 14,
        image_mean: Union[Tuple[float, float, float], List[float]] = (
            0.48145466,
            0.4578275,
            0.40821073,
        ),
        image_std: Union[Tuple[float, float, float], List[float]] = (
            0.26862954,
            0.26130258,
            0.27577711,
        ),
        rescale_factor: float = 1.0 / 255.0,
        do_normalize: bool = True,
        **kwargs,
    ):
        self.image_size = image_size
        self.min_size = min_size
        self.image_mean = image_mean
        self.image_std = image_std
        self.rescale_factor = rescale_factor
        self.do_normalize = do_normalize

        super().__init__(**kwargs)


class VLMImageProcessor(BaseImageProcessor):
    model_input_names = ["pixel_values"]

    def __init__(
        self,
        image_size: int,
        min_size: int = 14,
        image_mean: Union[Tuple[float, float, float], List[float]] = (
            0.48145466,
            0.4578275,
            0.40821073,
        ),
        image_std: Union[Tuple[float, float, float], List[float]] = (
            0.26862954,
            0.26130258,
            0.27577711,
        ),
        rescale_factor: float = 1.0 / 255.0,
        do_normalize: bool = True,
        **kwargs,
    ):
        super().__init__(**kwargs)

        self.image_size = image_size
        self.rescale_factor = rescale_factor
        self.image_mean = image_mean
        self.image_std = image_std
        self.min_size = min_size
        self.do_normalize = do_normalize

        if image_mean is None:
            self.background_color = (127, 127, 127)
        else:
            self.background_color = tuple([int(x * 255) for x in image_mean])
        self.background_color_tensor = torch.tensor(
            self.background_color, dtype=torch.float32
        ).view(3, 1, 1)
        self.image_mean_tensor = None
        self.image_std_tensor = None
        if self.image_mean is not None:
            self.image_mean_tensor = torch.tensor(
                self.image_mean, dtype=torch.float32
            ).view(1, -1, 1, 1)
        if self.image_std is not None:
            self.image_std_tensor = torch.tensor(
                self.image_std, dtype=torch.float32
            ).view(1, -1, 1, 1)

    def to_dict(self):
        """Serialize config fields without runtime tensor caches."""
        output = copy.deepcopy(self.__dict__)
        output.pop("background_color_tensor", None)
        output.pop("image_mean_tensor", None)
        output.pop("image_std_tensor", None)
        return output

    def _compute_resized_hw(self, height: int, width: int) -> Tuple[int, int]:
        max_size = max(width, height)
        new_height = max(int(height / max_size * self.image_size), self.min_size)
        new_width = max(int(width / max_size * self.image_size), self.min_size)

        if width <= 0 or height <= 0 or new_height <= 0 or new_width <= 0:
            raise ValueError(f"Invalid image size: {(height, width)} -> {(new_height, new_width)}")

        return new_height, new_width

    def _to_chw_tensor(self, image: ImageType) -> torch.Tensor:
        scale_unit_input = False

        if isinstance(image, Image.Image):
            if image.mode != "RGB":
                image = image.convert("RGB")
            tensor = torchvision.transforms.functional.pil_to_tensor(image)
        elif isinstance(image, np.ndarray):
            if image.ndim == 2:
                image = image[:, :, None]
            if image.ndim != 3:
                raise ValueError(f"Unsupported numpy image ndim: {image.ndim}")
            if image.shape[-1] in (1, 3, 4):
                tensor = torch.from_numpy(np.ascontiguousarray(image)).permute(2, 0, 1)
            elif image.shape[0] in (1, 3, 4):
                tensor = torch.from_numpy(np.ascontiguousarray(image))
            else:
                raise ValueError(f"Unsupported numpy image shape: {image.shape}")
        elif isinstance(image, torch.Tensor):
            tensor = image.detach().cpu()
            if tensor.ndim == 4 and tensor.shape[0] == 1:
                tensor = tensor.squeeze(0)
            if tensor.ndim == 2:
                tensor = tensor.unsqueeze(0)
            if tensor.ndim != 3:
                raise ValueError(f"Unsupported tensor image ndim: {tensor.ndim}")
            if tensor.shape[0] in (1, 3, 4):
                tensor = tensor.contiguous()
            elif tensor.shape[-1] in (1, 3, 4):
                tensor = tensor.permute(2, 0, 1).contiguous()
            else:
                raise ValueError(f"Unsupported tensor image shape: {tuple(tensor.shape)}")
        else:
            raise TypeError(f"Unsupported image type: {type(image)}")

        if tensor.shape[0] == 1:
            tensor = tensor.repeat(3, 1, 1)
        elif tensor.shape[0] == 4:
            tensor = tensor[:3]
        elif tensor.shape[0] != 3:
            raise ValueError(f"Unsupported channel count: {tensor.shape[0]}")

        scale_unit_input = tensor.is_floating_point()
        tensor = tensor.to(dtype=torch.float32)
        if scale_unit_input:
            max_value = tensor.max()
            min_value = tensor.min()
            if min_value >= 0.0 and max_value <= 1.0:
                tensor = tensor * 255.0

        return tensor

    def _resize_and_square_pad_tensor(self, image_tensor: torch.Tensor) -> torch.Tensor:
        _, height, width = image_tensor.shape
        new_height, new_width = self._compute_resized_hw(height, width)

        resized = torchvision.transforms.functional.resize(
            image_tensor,
            [new_height, new_width],
            interpolation=torchvision.transforms.functional.InterpolationMode.BICUBIC,
            antialias=True,
        )

        canvas = self.background_color_tensor.expand(
            -1, self.image_size, self.image_size
        ).clone()
        top = (self.image_size - new_height) // 2
        left = (self.image_size - new_width) // 2
        canvas[:, top : top + new_height, left : left + new_width] = resized

        return canvas

    def resize(self, pil_img: Image) -> np.ndarray:
        """

        Args:
            pil_img (PIL.Image): [H, W, 3] in PIL.Image in RGB

        Returns:
            x (np.ndarray): [3, self.image_size, self.image_size]
        """

        x = self._resize_and_square_pad_tensor(self._to_chw_tensor(pil_img))
        return x.clamp(0, 255).round().to(torch.uint8).cpu().numpy()

    def preprocess(self, images, return_tensors: str = "pt", **kwargs) -> BatchFeature:
        '''
        输入：images: List[np.ndarray]
        输出：BatchFeature({'pixel_values': tensor([...])})

        '''
        if isinstance(images, (Image.Image, np.ndarray, torch.Tensor)):
            images = [images]
        else:
            images = list(images)

        pixel_values = torch.stack(
            [self._resize_and_square_pad_tensor(self._to_chw_tensor(image)) for image in images],
            dim=0,
        ).contiguous()

        pixel_values.mul_(self.rescale_factor)

        if self.do_normalize and self.image_mean_tensor is not None and self.image_std_tensor is not None:
            pixel_values.sub_(self.image_mean_tensor).div_(self.image_std_tensor)

        if return_tensors is None:
            data = {"pixel_values": [image.cpu().numpy() for image in pixel_values]}
        else:
            data = {"pixel_values": pixel_values}
        return BatchFeature(data=data, tensor_type=return_tensors)

    @property
    def default_shape(self):
        return [3, self.image_size, self.image_size]


AutoImageProcessor.register(VLMImageProcessorConfig, VLMImageProcessor)


if __name__ == "__main__":
    image_processor = VLMImageProcessor(
        image_size=1024,
        image_mean=IMAGENET_INCEPTION_MEAN,
        image_std=IMAGENET_INCEPTION_STD,
        do_normalize=True,
    )
