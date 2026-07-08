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

from dataclasses import dataclass
from typing import Dict, List

import torch
from PIL.Image import Image
from transformers import LlamaTokenizerFast
from transformers.processing_utils import ProcessorMixin

from janus.models.image_processing_vlm import VLMImageProcessor
from janus.utils.conversation import get_conv_template
import copy


def _normalize_extra_special_tokens(extra_special_tokens):
    if extra_special_tokens is None:
        return []
    if isinstance(extra_special_tokens, str):
        raw = extra_special_tokens.strip()
        if not raw:
            return []
        parts = raw.split(",") if "," in raw else raw.split()
        return [part.strip() for part in parts if part.strip()]
    return [str(token).strip() for token in extra_special_tokens if str(token).strip()]


def _dedupe_preserve_order(tokens):
    seen = set()
    deduped = []
    for token in tokens:
        if token in seen:
            continue
        seen.add(token)
        deduped.append(token)
    return deduped


def _filter_extra_special_tokens(extra_special_tokens, existing_special_tokens=None):
    existing_special_tokens = set(existing_special_tokens or [])
    return [
        token for token in _dedupe_preserve_order(
            _normalize_extra_special_tokens(extra_special_tokens)
        )
        if token not in existing_special_tokens
    ]


def _add_additional_special_tokens(tokenizer, tokens):
    if not tokens:
        return 0
    special_tokens_dict = {"additional_special_tokens": list(tokens)}
    try:
        return tokenizer.add_special_tokens(
            special_tokens_dict,
            replace_additional_special_tokens=False,
        )
    except TypeError:
        existing = list(getattr(tokenizer, "additional_special_tokens", []))
        merged = _dedupe_preserve_order([*existing, *tokens])
        return tokenizer.add_special_tokens({"additional_special_tokens": merged})


def _add_extra_special_tokens_to_tokenizer(tokenizer, extra_special_tokens, existing_special_tokens=None):
    tokens = _filter_extra_special_tokens(
        extra_special_tokens,
        existing_special_tokens=existing_special_tokens,
    )
    _add_additional_special_tokens(tokenizer, tokens)
    return tokens

class DictOutput(object):
    def keys(self):
        return self.__dict__.keys()

    def __getitem__(self, item):
        return self.__dict__[item]

    def __setitem__(self, key, value):
        self.__dict__[key] = value


@dataclass
class VLChatProcessorOutput(DictOutput):
    sft_format: str
    input_ids: torch.Tensor
    pixel_values: torch.Tensor
    num_image_tokens: torch.IntTensor

    def __len__(self):
        return len(self.input_ids)


@dataclass
class BatchedVLChatProcessorOutput(DictOutput):
    sft_format: List[str]
    input_ids: torch.Tensor
    pixel_values: torch.Tensor
    attention_mask: torch.Tensor
    images_seq_mask: torch.BoolTensor
    images_emb_mask: torch.BoolTensor

    def to(self, device, dtype=torch.bfloat16):
        self.input_ids = self.input_ids.to(device)
        self.attention_mask = self.attention_mask.to(device)
        self.images_seq_mask = self.images_seq_mask.to(device)
        self.images_emb_mask = self.images_emb_mask.to(device)
        self.pixel_values = self.pixel_values.to(device=device, dtype=dtype)
        return self


class VLChatProcessor(ProcessorMixin):
    '''
    Args:
        image_processor: 图像处理器
        tokenizer: 分词器
        image_tag: 图像标签
        image_start_tag: 图像开始标签
        image_end_tag: 图像结束标签
    Returns:
        None
    '''
    image_processor_class = "AutoImageProcessor"
    tokenizer_class = ("LlamaTokenizer", "LlamaTokenizerFast")


    attributes = ["image_processor", "tokenizer"]

    system_prompt = (
        "You are a helpful language and vision assistant. "
        "You are able to understand the visual content that the user provides, "
        "and assist the user with a variety of tasks using natural language."
    )

    def __init__(
        self,
        image_processor: VLMImageProcessor,#加载图像处理器
        tokenizer: LlamaTokenizerFast,#加载分词器
        image_tag: str = "<image_placeholder>",
        image_start_tag: str = "<begin_of_image>",
        image_end_tag: str = "<end_of_image>",
        pad_tag: str = "<｜▁pad▁｜>",
        num_image_tokens: int = 576,
        add_special_token: bool = False,
        sft_format: str = "deepseek",
        mask_prompt: bool = True,
        ignore_id: int = -100,
        **kwargs,
    ):
        extra_special_tokens = _normalize_extra_special_tokens(kwargs.pop("extra_special_tokens", None))
        skip_output_special_tokens = bool(kwargs.pop("skip_output_special_tokens", False))
        '''
        print("VLChatProcessor init")
        print("image_processor: ", image_processor)
        #print("tokenizer: ", tokenizer)
        print("image_tag: ", image_tag)
        print("image_start_tag: ", image_start_tag)
        print("image_end_tag: ", image_end_tag)
        print("pad_tag: ", pad_tag)
        print("num_image_tokens: ", num_image_tokens)
        print("add_special_token: ", add_special_token)
        print("sft_format: ", sft_format)
        print("mask_prompt: ", mask_prompt)
        print("ignore_id: ", ignore_id)
        print("kwargs: ", kwargs)
        '''
        self.image_processor = image_processor
        self.tokenizer = tokenizer

        self.original_tokenizer_len = len(self.tokenizer)
        special_tokens = []
        image_id = self.tokenizer.vocab.get(image_tag)
        if image_id is None:
            if skip_output_special_tokens:
                raise ValueError(
                    f"{image_tag!r} is missing from the base tokenizer and "
                    "skip_output_special_tokens=True forbids tokenizer expansion."
                )
            special_tokens = [image_tag]
        if not skip_output_special_tokens:
            special_tokens.extend(["<|latent_start|>", "<|latent_pad|>", "<|latent_end|>"])
            if image_id is not None:
                special_tokens.append(f'</MOVE>')
                special_tokens.append(f'</PICK>')
                special_tokens.append(f'</PLACE>')
                special_tokens.append(f'</ROTATE>')
                special_tokens.append(f'</PULL>')
                special_tokens.append(f'</PUSH>')
                special_tokens.append(f'</NONE>')

        existing_special_tokens = set(special_tokens)
        if skip_output_special_tokens:
            self.extra_special_tokens = []
        else:
            self.extra_special_tokens = _filter_extra_special_tokens(
                extra_special_tokens,
                existing_special_tokens=existing_special_tokens,
            )
        special_tokens.extend(self.extra_special_tokens)
        special_tokens = _dedupe_preserve_order(special_tokens)
        #print("special_tokens: ", special_tokens)
        #加进去了一些特殊token
        self.num_add_tokens = _add_additional_special_tokens(self.tokenizer, special_tokens)
        #print(f"Add {self.num_add_tokens} spectial token to the tokenizer!!")

        self.image_tag = image_tag
        self.image_start_tag = image_start_tag
        self.image_end_tag = image_end_tag
        self.pad_tag = pad_tag

        self.num_image_tokens = num_image_tokens
        self.add_special_token = add_special_token
        self.sft_format = sft_format
        self.mask_prompt = mask_prompt
        self.ignore_id = ignore_id
        self.skip_output_special_tokens = skip_output_special_tokens

        super().__init__(
            image_processor,
            tokenizer,
            image_tag,
            num_image_tokens,
            add_special_token,
            sft_format,
            mask_prompt,
            ignore_id,
            **kwargs,
        )

        # # === DEBUG: special tokens ===
        # print("\n========== [DEBUG] Tokenizer Special Tokens ==========")
        # print(f"Original vocab size: {self.original_tokenizer_len}")
        # print(f"Added vocab size:    {len(self.tokenizer)}")
        # print(f"Actually added {self.num_add_tokens} special tokens")

        # all_specials = self.tokenizer.additional_special_tokens
        # print("\nAll special tokens:")
        # for i, tok in enumerate(all_specials):
        #     tid = self.tokenizer.convert_tokens_to_ids(tok)
        #     print(f"  {i:3d}: {tok:<25} -> id={tid}")

        # critical_tokens = [
        #     self.image_tag,
        #     self.image_start_tag,
        #     self.image_end_tag,
        #     "<|latent_start|>",
        #     "<|latent_pad|>",
        #     "<|latent_end|>",
        # ]
        # print("\nCritical token existence check:")
        # for t in critical_tokens:
        #     tid = self.tokenizer.convert_tokens_to_ids(t)
        #     print(f"  {t:<25} -> id={tid}")

        # print("=======================================================\n")
        # input("tokenizer check done, press any key to continue")

    @classmethod
    def from_pretrained(
        cls,
        pretrained_model_name_or_path,
        *args,
        extra_special_tokens=None,
        skip_output_special_tokens=False,
        **kwargs,
    ):
        if skip_output_special_tokens:
            kwargs["skip_output_special_tokens"] = True
        processor = super().from_pretrained(pretrained_model_name_or_path, *args, **kwargs)
        if extra_special_tokens is not None and not skip_output_special_tokens:
            processor.add_extra_special_tokens(extra_special_tokens)
        return processor

    def add_extra_special_tokens(self, extra_special_tokens):
        if getattr(self, "skip_output_special_tokens", False):
            return []
        base_special_tokens = [
            "<|latent_start|>",
            "<|latent_pad|>",
            "<|latent_end|>",
            "</MOVE>",
            "</PICK>",
            "</PLACE>",
            "</ROTATE>",
            "</PULL>",
            "</PUSH>",
            "</NONE>",
            *getattr(self, "extra_special_tokens", []),
        ]
        new_tokens = _add_extra_special_tokens_to_tokenizer(
            self.tokenizer,
            extra_special_tokens,
            existing_special_tokens=base_special_tokens,
        )
        self.extra_special_tokens = [
            *getattr(self, "extra_special_tokens", []),
            *new_tokens,
        ]
        return new_tokens

    def new_chat_template(self):
        '''
        返回一个对话模板，并且设置系统提示语
        '''
        conv = get_conv_template(self.sft_format)
        conv.set_system_message(self.system_prompt)
        return conv

    def apply_sft_template_for_multi_turn_prompts(
        self,
        conversations: List[Dict[str, str]],
        sft_format: str = "deepseek",
        system_prompt: str = "",
    ):
        """
        Applies the SFT template to conversation.

        An example of conversation:
        conversation = [
            {
                "role": "User",
                "content": "<image_placeholder> is Figure 1.\n<image_placeholder> is Figure 2.\nWhich image is brighter?",
                "images": [
                    "./multi-images/attribute_comparison_1.png",
                    "./multi-images/attribute_comparison_2.png"
                ]
            },
            {
                "role": "Assistant",
                "content": ""
            }
        ]

        Args:
            conversations (List[Dict]): A conversation with a List of Dict[str, str] text.
            sft_format (str, optional): The format of the SFT template to use. Defaults to "deepseek".
            system_prompt (str, optional): The system prompt to use in the SFT template. Defaults to "".

        Returns:
            sft_prompt (str): The formatted text.
        """

        conv = get_conv_template(sft_format)
        conv.set_system_message(system_prompt)
        #print("conversations: ", conversations)
        for message in conversations:
            conv.append_message(message["role"], message["content"].strip())
        sft_prompt = conv.get_prompt().strip()
        #print("sft_prompt: ", sft_prompt)
        return sft_prompt

    @property
    def image_token(self):
        return self.image_tag

    @property
    def image_id(self):
        image_id = self.tokenizer.vocab.get(self.image_tag)
        return image_id

    @property
    def image_start_id(self):
        image_start_id = self.tokenizer.vocab.get(self.image_start_tag)
        return image_start_id

    @property
    def image_end_id(self):
        image_end_id = self.tokenizer.vocab.get(self.image_end_tag)
        return image_end_id

    @property
    def image_start_token(self):
        return self.image_start_tag

    @property
    def image_end_token(self):
        return self.image_end_tag

    @property
    def pad_id(self):
        pad_id = self.tokenizer.vocab.get(self.pad_tag)
        return pad_id

    def add_image_token(
        self,
        image_indices: List[int],
        input_ids: torch.LongTensor,
    ):
        """
        输入：图像索引的所有应该插入位置的索引
            以及原来的input_ids
        输出：插入图像token后的input_ids
            以及每个图像的token数量列表（tensor类型）

        Args:
            image_indices (List[int]): [index_0, index_1, ..., index_j]
            input_ids (torch.LongTensor): [N]

        Returns:
            input_ids (torch.LongTensor): [N + image tokens]
            num_image_tokens (torch.IntTensor): [n_images]
        """

        input_slices = []

        start = 0
        for index in image_indices:
            if self.add_special_token:
                end = index + 1
            else:
                end = index

            # original text tokens
            input_slices.append(input_ids[start:end])

            # add boi, image tokens, eoi and set the mask as False
            input_slices.append(self.image_start_id * torch.ones((1), dtype=torch.long))
            input_slices.append(
                self.image_id * torch.ones((self.num_image_tokens,), dtype=torch.long)
            )
            input_slices.append(self.image_end_id * torch.ones((1), dtype=torch.long))
            start = index + 1

        # the left part
        input_slices.append(input_ids[start:])

        # concat all slices
        input_ids = torch.cat(input_slices, dim=0)
        num_image_tokens = torch.IntTensor([self.num_image_tokens] * len(image_indices))

        return input_ids, num_image_tokens

    def process_one(
        self,
        prompt: str = None,
        conversations: List[Dict[str, str]] = None,
        images: List[Image] = None,
        add_gen_image_token: bool = False,
        **kwargs,
    ):
        """

        Args:
            prompt (str): the formatted prompt;
            conversations (List[Dict]): conversations with a list of messages;
            images (List[ImageType]): the list of images;
            **kwargs:

        Returns:
            outputs (BaseProcessorOutput): the output of the processor,
                - input_ids (torch.LongTensor): [N + image tokens]
                - target_ids (torch.LongTensor): [N + image tokens]
                - images (torch.FloatTensor): [n_images, 3, H, W]
                - image_id (int): the id of the image token
                - num_image_tokens (List[int]): the number of image tokens
        """

        assert (
            prompt is None or conversations is None
        ), "prompt and conversations cannot be used at the same time."

        if prompt is None:
            # apply sft format
            sft_format = self.apply_sft_template_for_multi_turn_prompts(
                conversations=conversations,
                sft_format=self.sft_format,
                system_prompt=self.system_prompt,
            )
        else:
            sft_format = prompt
        #print("sft_format: ", sft_format)
        
        if add_gen_image_token:
            sft_format += self.image_start_tag

        # tokenize
        input_ids = self.tokenizer.encode(sft_format)
        input_ids = torch.LongTensor(input_ids)
        # add image tokens to the input_ids
        image_token_mask: torch.BoolTensor = input_ids == self.image_id
        image_indices = image_token_mask.nonzero()
        input_ids, num_image_tokens = self.add_image_token(
            image_indices=image_indices,
            input_ids=input_ids,
        )

        # load images
        images_outputs = self.image_processor(images, return_tensors="pt")

        prepare = VLChatProcessorOutput(
            sft_format=sft_format,
            input_ids=input_ids,
            pixel_values=images_outputs.pixel_values,
            num_image_tokens=num_image_tokens,
        )

        return prepare

    def __call__(
        self,
        *,
        prompt: str = None,
        conversations: List[Dict[str, str]] = None,
        images: List[Image] = None,
        force_batchify: bool = True,
        add_gen_image_token: bool = False,
        **kwargs,
    ):
        """

        Args:
            prompt (str): the formatted prompt;
            conversations (List[Dict]): conversations with a list of messages;
            images (List[ImageType]): the list of images;
            force_batchify (bool): force batchify the inputs;
            **kwargs:

        Returns:
            outputs (BaseProcessorOutput): the output of the processor,
                - input_ids (torch.LongTensor): [N + image tokens]
                - images (torch.FloatTensor): [n_images, 3, H, W]
                - image_id (int): the id of the image token
                - num_image_tokens (List[int]): the number of image tokens
        """

        prepare = self.process_one(
            prompt=prompt, conversations=conversations, images=images, add_gen_image_token=add_gen_image_token
        )

        if force_batchify:
            prepare = self.batchify([prepare])

        return prepare

    def get_imgtext2image_prepare_inputs(
        self,
        *,
        prompt: str = None,
        conversations: List[Dict[str, str]] = None,
        images: List[Image] = None,
        force_batchify: bool = True,
        add_gen_image_token: bool = False,
        **kwargs,
    ):
        """

        Args:
            prompt (str): the formatted prompt;
            conversations (List[Dict]): conversations with a list of messages;
            images (List[ImageType]): the list of images;
            force_batchify (bool): force batchify the inputs;
            **kwargs:

        Returns:
            outputs (BaseProcessorOutput): the output of the processor,
                - input_ids (torch.LongTensor): [N + image tokens]
                - images (torch.FloatTensor): [n_images, 3, H, W]
                - image_id (int): the id of the image token
                - num_image_tokens (List[int]): the number of image tokens
        """

        prepare = self.process_one(
            prompt=prompt, conversations=conversations, images=images, add_gen_image_token=add_gen_image_token
        )
        # make another empty prepare for the cfg
        prepare_empty = copy.deepcopy(prepare) # now, we empty all the prompt for cfg
        prepare_empty.input_ids[1:-1] = self.pad_id

        if force_batchify:
            prepare = self.batchify([prepare])

        return prepare

    def batchify(
        self, prepare_list: List[VLChatProcessorOutput]
    ) -> BatchedVLChatProcessorOutput:
        """
        Preprocesses the inputs for multimodal inference.

        Args:
            prepare_list (List[VLChatProcessorOutput]): A list of VLChatProcessorOutput.

        Returns:
            BatchedVLChatProcessorOutput: A dictionary of the inputs to use for multimodal inference.
        """

        batch_size = len(prepare_list)
        sft_format = []
        n_images = []
        seq_lens = []
        for prepare in prepare_list:
            n_images.append(len(prepare.num_image_tokens))
            seq_lens.append(len(prepare))

        input_token_max_len = max(seq_lens)
        max_n_images = max(1, max(n_images))

        batched_input_ids = torch.full(
            (batch_size, input_token_max_len), self.pad_id
        ).long()  # FIXME
        batched_attention_mask = torch.zeros((batch_size, input_token_max_len)).long()
        batched_pixel_values = torch.zeros(
            (batch_size, max_n_images, *self.image_processor.default_shape)
        ).float()
        batched_images_seq_mask = torch.zeros((batch_size, input_token_max_len)).bool()
        batched_images_emb_mask = torch.zeros(
            (batch_size, max_n_images, self.num_image_tokens)
        ).bool()

        for i, prepare in enumerate(prepare_list):
            input_ids = prepare.input_ids
            seq_len = len(prepare)
            n_image = len(prepare.num_image_tokens)
            # left-padding
            batched_attention_mask[i, -seq_len:] = 1
            batched_input_ids[i, -seq_len:] = torch.LongTensor(input_ids)
            batched_images_seq_mask[i, -seq_len:] = input_ids == self.image_id

            if n_image > 0:
                batched_pixel_values[i, :n_image] = prepare.pixel_values
                for j, n_image_tokens in enumerate(prepare.num_image_tokens):
                    batched_images_emb_mask[i, j, :n_image_tokens] = True

            sft_format.append(prepare.sft_format)

        batched_prepares = BatchedVLChatProcessorOutput(
            input_ids=batched_input_ids,
            attention_mask=batched_attention_mask,
            pixel_values=batched_pixel_values,
            images_seq_mask=batched_images_seq_mask,
            images_emb_mask=batched_images_emb_mask,
            sft_format=sft_format,
        )

        return batched_prepares
    
    def batchify_gen(
        self, prepare_list: List[VLChatProcessorOutput]
    ) -> BatchedVLChatProcessorOutput:
        """
        The only difference between batchify and batchify_gen is the padding direction. (right here)
        Preprocesses the inputs for multimodal generation.

        Args:
            prepare_list (List[VLChatProcessorOutput]): A list of VLChatProcessorOutput.

        Returns:
            BatchedVLChatProcessorOutput: A dictionary of the inputs to use for multimodal inference.
        """

        batch_size = len(prepare_list)
        sft_format = []
        n_images = []
        seq_lens = []
        for prepare in prepare_list:
            n_images.append(len(prepare.num_image_tokens))
            seq_lens.append(len(prepare))

        input_token_max_len = max(seq_lens)
        max_n_images = max(1, max(n_images))

        batched_input_ids = torch.full(
            (batch_size, input_token_max_len), self.pad_id
        ).long()  # FIXME
        batched_attention_mask = torch.zeros((batch_size, input_token_max_len)).long()
        batched_pixel_values = torch.zeros(
            (batch_size, max_n_images, *self.image_processor.default_shape)
        ).float()
        batched_images_seq_mask = torch.zeros((batch_size, input_token_max_len)).bool()
        batched_images_emb_mask = torch.zeros(
            (batch_size, max_n_images, self.num_image_tokens)
        ).bool()

        for i, prepare in enumerate(prepare_list):
            input_ids = prepare.input_ids
            seq_len = len(prepare)
            n_image = len(prepare.num_image_tokens)
            # right-padding
            batched_attention_mask[i, :seq_len] = 1
            batched_input_ids[i, :seq_len] = torch.LongTensor(input_ids)
            batched_images_seq_mask[i, :seq_len] = input_ids == self.image_id

            if n_image > 0:
                batched_pixel_values[i, :n_image] = prepare.pixel_values
                for j, n_image_tokens in enumerate(prepare.num_image_tokens):
                    batched_images_emb_mask[i, j, :n_image_tokens] = True

            sft_format.append(prepare.sft_format)

        batched_prepares = BatchedVLChatProcessorOutput(
            input_ids=batched_input_ids,
            attention_mask=batched_attention_mask,
            pixel_values=batched_pixel_values,
            images_seq_mask=batched_images_seq_mask,
            images_emb_mask=batched_images_emb_mask,
            sft_format=sft_format,
        )

        return batched_prepares
