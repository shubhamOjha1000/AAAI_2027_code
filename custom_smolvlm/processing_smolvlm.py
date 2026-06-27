# Copyright 2025 The HuggingFace Inc. team.
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
"""
Processor class for SmolVLM.
"""

from datetime import timedelta
from typing import TYPE_CHECKING, Union

from ...feature_extraction_utils import BatchFeature
from ...image_utils import ImageInput, make_nested_list_of_images
from ...processing_utils import ProcessingKwargs, ProcessorMixin, Unpack
from ...tokenization_utils_base import BatchEncoding, TextInput
from ...utils import auto_docstring as _auto_docstring_real, is_num2words_available, logging

def auto_docstring(*args, **kwargs):
    try:
        result = _auto_docstring_real(*args, **kwargs)
    except (ValueError, Exception):
        if args and callable(args[0]):
            return args[0]
        return lambda cls: cls
    if args and callable(args[0]):
        return result
    def _safe_apply(cls):
        try:
            return result(cls)
        except (ValueError, Exception):
            return cls
    return _safe_apply
from ...video_utils import VideoInput

print("[custom_smolvlm] processing_smolvlm.py loaded — CUSTOM PROCESSOR ACTIVE")


# Adapted from transformers.models.smolvlm.video_processing_smolvlm.DEFAULT_VIDEO_INTRO
DEFAULT_VIDEO_INTRO = (
    "You are provided the following series of {frame_count} frames from a {video_duration} [H:MM:SS] video.\n"
)
# Adapted from transformers.models.smolvlm.video_processing_smolvlm.DEFAULT_MEDIA_OUTTRO
DEFAULT_MEDIA_OUTTRO = "\n\n"
# Adapted from transformers.models.smolvlm.video_processing_smolvlm.FRAME_TIMESTAMP_MESSAGE
FRAME_TIMESTAMP_MESSAGE = "\nFrame from {timestamp}:"

if TYPE_CHECKING:
    from ...tokenization_utils_base import PreTokenizedInput

logger = logging.get_logger(__name__)


if is_num2words_available():
    from num2words import num2words
else:
    num2words = None


# The correct chat template to be used for videos after #38105
DEFAULT_CHAT_TEMPLATE = "<|im_start|>{% for message in messages %}{{message['role'] | capitalize}}{% if message['content'][0]['type'] == 'image' %}{{':'}}{% else %}{{': '}}{% endif %}{% for line in message['content'] %}{% if line['type'] == 'text' %}{{line['text']}}{% elif line['type'] == 'image' %}{{ '<image>' }}{% elif line['type'] == 'video' %}{{ '<video>' }}{% endif %}{% endfor %}<end_of_utterance>\n{% endfor %}{% if add_generation_prompt %}{{ 'Assistant:' }}{% endif %}"


def _prompt_split_image(
    image_seq_len, image_rows, image_cols, fake_token_around_image, image_token, global_image_token
):
    """Prompt with expanded image tokens for when the image is split into patches."""
    text_split_images = ""
    for n_h in range(image_rows):
        for n_w in range(image_cols):
            text_split_images += (
                f"{fake_token_around_image}" + f"<row_{n_h + 1}_col_{n_w + 1}>" + f"{image_token}" * image_seq_len
            )
        text_split_images += "\n"

    text_split_images += (
        f"\n{fake_token_around_image}"
        + f"{global_image_token}"
        + f"{image_token}" * image_seq_len
        + f"{fake_token_around_image}"
    )
    return text_split_images


def _prompt_single_image(image_seq_len, fake_token_around_image, image_token, global_image_token):
    """Prompt with expanded image tokens for a single image."""
    return (
        f"{fake_token_around_image}"
        + f"{global_image_token}"
        + f"{image_token}" * image_seq_len
        + f"{fake_token_around_image}"
    )


def get_image_prompt_string(
    image_rows, image_cols, image_seq_len, fake_token_around_image, image_token, global_image_token
):
    if image_rows == 0 and image_cols == 0:
        return _prompt_single_image(
            image_seq_len,
            fake_token_around_image=fake_token_around_image,
            image_token=image_token,
            global_image_token=global_image_token,
        )
    return _prompt_split_image(
        image_seq_len, image_rows, image_cols, fake_token_around_image, image_token, global_image_token
    )


def gaze_to_tiles(gaze, image_size, R, C, n_img, tau: float = 0.15, neighbors: bool = True):
    """Gaze-driven native tile selection (Stage 1 of foveated_tiling_spec.md).

    Map a gaze point to the tile index it falls in and return the sub-images to
    KEEP: that tile, optional straddle-neighbors when the gaze sits within ``tau``
    of a tile boundary, plus the global thumbnail (always the LAST sub-image).

    Args:
        gaze: ``(gx, gy)``. Values in ``[0, 1]`` are treated as NORMALIZED
            fractions; values ``> 1`` as absolute pixels. Off-screen / invalid
            gaze -> global-only fallback ``[n_img - 1]``.
        image_size: ``(W, H)`` of the frame the gaze refers to (used only to
            normalize pixel-coordinate gaze).
        R, C: tile grid (rows x cols) chosen by the processor.
        n_img: number of sub-images = ``R * C + 1`` (tiles + 1 global).
        tau: edge margin (fraction of a cell) for adding straddle-neighbors.
        neighbors: if False, never add neighbors (gaze tile + global only).

    Returns:
        Sorted list of kept sub-image indices into ``pixel_values[0]``
        (tiles are row-major ``0 .. R*C-1``; global is ``n_img - 1``).
    """
    W, H = image_size
    gx, gy = float(gaze[0]), float(gaze[1])
    # Normalized [0,1] -> use as-is; otherwise treat as pixels and normalize.
    if 0.0 <= gx <= 1.0 and 0.0 <= gy <= 1.0:
        fx, fy = gx, gy
    else:
        fx = gx / W if W else gx
        fy = gy / H if H else gy

    global_idx = n_img - 1
    # Off-screen / invalid gaze -> global-only fallback.
    if not (0.0 <= fx <= 1.0 and 0.0 <= fy <= 1.0):
        return [global_idx]

    col = min(int(fx * C), C - 1)
    row = min(int(fy * R), R - 1)
    gaze_tile = row * C + col
    keep = {gaze_tile}

    if neighbors:
        in_col = fx * C - col          # position within the cell, x (0..1)
        in_row = fy * R - row          # position within the cell, y (0..1)
        if in_col < tau and col > 0:           keep.add(row * C + col - 1)   # left
        if in_col > 1 - tau and col < C - 1:   keep.add(row * C + col + 1)   # right
        if in_row < tau and row > 0:           keep.add((row - 1) * C + col) # top
        if in_row > 1 - tau and row < R - 1:   keep.add((row + 1) * C + col) # bottom

    keep.add(global_idx)
    return sorted(keep)


class SmolVLMProcessorKwargs(ProcessingKwargs, total=False):
    _defaults = {
        "text_kwargs": {
            "add_special_tokens": True,
            "padding": False,
            "is_split_into_words": False,
        },
        "images_kwargs": {
            "return_row_col_info": True,
        },
        "videos_kwargs": {
            "return_metadata": True,
        },
    }


@auto_docstring
class SmolVLMProcessor(ProcessorMixin):
    def __init__(
        self,
        image_processor,
        tokenizer,
        video_processor=None,
        image_seq_len: int = 169,
        chat_template: str | None = None,
        **kwargs,
    ):
        r"""
        image_seq_len (`int`, *optional*, defaults to 169):
            The length of the image sequence i.e. the number of <image> tokens per image in the input.
            This parameter is used to build the string from the input prompt and image tokens and should match the
            value the model used. It is computed as: image_seq_len = int(((image_size // patch_size) ** 2) / (scale_factor**2))
        """
        self.fake_image_token = getattr(tokenizer, "fake_image_token", "<fake_token_around_image>")
        self.image_token = getattr(tokenizer, "image_token", "<image>")
        self.image_token_id = tokenizer.convert_tokens_to_ids(self.image_token)
        self.end_of_utterance_token = getattr(tokenizer, "end_of_utterance_token", "<end_of_utterance>")
        self.global_image_token = getattr(tokenizer, "global_image_token", "<global-img>")
        self.image_seq_len = image_seq_len
        self.video_token = getattr(tokenizer, "video_token", "<video>")

        if not num2words:
            import warnings
            warnings.warn(
                "Package `num2words` is not installed. Video prompt timestamps will be unavailable. "
                "Install with `pip install num2words` for full functionality."
            )

        super().__init__(image_processor, tokenizer, video_processor, chat_template=chat_template, **kwargs)

    def expand_text_with_image_tokens(self, text, image_rows, image_cols):
        prompt_strings = []
        for sample, sample_rows, sample_cols in zip(text, image_rows, image_cols):
            # Replace the image token with fake tokens around the expanded image token sequence of length `image_seq_len`
            image_prompt_strings = []
            for n_rows, n_cols in zip(sample_rows, sample_cols):
                image_prompt_string = get_image_prompt_string(
                    n_rows,
                    n_cols,
                    self.image_seq_len,
                    image_token=self.image_token,
                    fake_token_around_image=self.fake_image_token,
                    global_image_token=self.global_image_token,
                )
                image_prompt_strings.append(image_prompt_string)

            split_sample = sample.split(self.image_token)
            if len(split_sample) == 0:
                raise ValueError("The image token should be present in the text.")

            # Place in the image prompt strings where the image tokens are
            sample = split_sample[0]
            for i, image_prompt_string in enumerate(image_prompt_strings):
                sample += image_prompt_string + split_sample[i + 1]
            prompt_strings.append(sample)

        return prompt_strings

    # ------------------------------------------------------------------ #
    #  Gaze-driven foveated tiling (native tile selection)                #
    #  See foveated_tiling_spec.md. Training-free; in-distribution.       #
    # ------------------------------------------------------------------ #
    def gaze_to_keep(self, gaze, image_size, R, C, n_img, tau: float = 0.15, neighbors: bool = True):
        """Stage 1 wrapper: gaze -> kept sub-image indices (see ``gaze_to_tiles``)."""
        return gaze_to_tiles(gaze, image_size, R, C, n_img, tau=tau, neighbors=neighbors)

    def _build_foveated_text(self, text, keep, R, C, n_img):
        """Stage 2 (text side): rebuild the image-prompt with ONLY the kept tiles.

        Each kept tile keeps its real ``<row_r_col_c>`` tag (in-distribution); the
        global block is always appended last. The dropped tiles' token blocks are
        removed so ``count(<image>) == len(keep) * image_seq_len``.
        """
        global_idx = n_img - 1
        tiles = [k for k in keep if k != global_idx]
        seq = self.image_seq_len
        img, fake, glob = self.image_token, self.fake_image_token, self.global_image_token

        block = ""
        last_row = None
        for t in tiles:
            r, c = t // C, t % C
            if last_row is not None and r != last_row:
                block += "\n"
            block += f"{fake}<row_{r + 1}_col_{c + 1}>" + img * seq
            last_row = r

        if tiles:
            block += f"\n\n{fake}{glob}" + img * seq + f"{fake}"
        else:  # global-only (no tiles kept) -> single-image prompt format
            block += f"{fake}{glob}" + img * seq + f"{fake}"

        split_sample = text.split(img)
        if len(split_sample) < 2:
            raise ValueError("The image token should be present in the text.")
        # One <image> placeholder -> two parts; join defensively if more.
        return split_sample[0] + block + img.join(split_sample[1:])

    def build_foveated_inputs(
        self, image, text, gaze, tau: float = 0.15, neighbors: bool = True,
        return_tensors: str = "pt", verbose: bool = False,
    ):
        """Build foveated model inputs for one ``(image, prompt, gaze)``.

        Implements Stages 1-2 of foveated_tiling_spec.md with **native tile
        selection**: run the standard tiling, then keep only the gaze tile(s) +
        global and DROP the rest from BOTH ``pixel_values`` and ``input_ids`` (real
        compute / KV-cache savings — not masking). Stage 3 (encode/decode) is the
        stock ``model.generate(**inputs)``.

        Args:
            image: a single PIL image / array / path.
            text:  prompt containing exactly one ``<image>`` placeholder
                   (e.g. from ``apply_chat_template``).
            gaze:  ``(gx, gy)`` normalized ([0,1]) or pixels (>1); off-screen ->
                   global-only fallback.
            tau / neighbors: straddle-neighbor controls (see ``gaze_to_tiles``).

        Returns:
            ``dict`` with ``pixel_values``, ``input_ids``, ``attention_mask``
            (and ``pixel_attention_mask`` if padded), plus ``keep``, ``grid``
            ``(R, C)`` and ``n_partitions`` metadata.
        """
        images = make_nested_list_of_images(self.image_processor.fetch_images([image]))
        pil = images[0][0]
        size = getattr(pil, "size", None)        # PIL: (W, H); used only for pixel gaze
        W, H = size if size is not None else (None, None)

        vision = self.image_processor(images, return_row_col_info=True, return_tensors=return_tensors)
        pixel_values = vision["pixel_values"]    # [1, N_img, 3, h, w]
        n_img = pixel_values.shape[1]
        R = int(vision["rows"][0][0])
        C = int(vision["cols"][0][0])

        if R == 0 or C == 0:
            # Image was not split -> only the global image exists; nothing to drop.
            keep = [n_img - 1]
        else:
            keep = gaze_to_tiles(gaze, (W, H), R, C, n_img, tau=tau, neighbors=neighbors)

        pixel_values_fov = pixel_values[:, keep]
        fov_text = self._build_foveated_text(text, keep, R, C, n_img)
        tok = self.tokenizer(fov_text, return_tensors=return_tensors)

        data = {
            "pixel_values": pixel_values_fov,
            "input_ids": tok["input_ids"],
            "attention_mask": tok["attention_mask"],
        }
        if "pixel_attention_mask" in vision:
            data["pixel_attention_mask"] = vision["pixel_attention_mask"][:, keep]

        # THE INVARIANT (crash point): <image> count must match kept sub-images.
        try:
            n_img_tok = int((tok["input_ids"] == self.image_token_id).sum())
            expected = len(keep) * self.image_seq_len
            assert n_img_tok == expected, (
                f"foveation invariant broken: {n_img_tok} <image> tokens != {expected}"
            )
        except TypeError:
            pass  # non-tensor return_tensors; skip the tensor-only check

        if verbose:
            print(f"[foveate] grid={R}x{C}  N_img={n_img}  keep={keep}  "
                  f"({len(keep)} partitions, {len(keep) * self.image_seq_len} visual tokens; "
                  f"full={n_img * self.image_seq_len})")

        return {"keep": keep, "grid": (R, C), "n_partitions": len(keep), **data}

    def expand_text_with_video_tokens(self, text, video_inputs):
        num_frames = video_inputs["pixel_values"].shape[1]
        video_metadata = iter(video_inputs["video_metadata"])

        prompt_strings = []
        for sample in text:
            while self.video_token in sample:
                metadata = next(video_metadata)
                if metadata.fps is None:
                    logger.warning_once(
                        "SmolVLM requires frame timestamps to construct prompts, but the `fps` of the input video could not be inferred. "
                        "Probably `video_metadata` was missing from inputs and you passed pre-sampled frames. "
                        "Defaulting to `fps=24`. Please provide `video_metadata` for more accurate results."
                    )
                    metadata.fps = 24  # Set the default fps to 24 for BC, otherwise `timestamps` can't be inferred
                timestamps = [(int(second // 60), int(second % 60)) for second in metadata.timestamps]
                duration = int(metadata.duration) if metadata.duration is not None else int(metadata.timestamps[-1])
                duration_td = timedelta(seconds=int(duration))
                image_prompt_strings = DEFAULT_VIDEO_INTRO.format(
                    frame_count=num2words(num_frames), video_duration=str(duration_td)
                )
                for timestamp in timestamps:
                    image_prompt_string = _prompt_single_image(
                        self.image_seq_len,
                        image_token=self.image_token,
                        fake_token_around_image=self.fake_image_token,
                        global_image_token=self.global_image_token,
                    )
                    timestamp = f"{timestamp[0]:02d}:{timestamp[1]:02d}"
                    image_prompt_string = FRAME_TIMESTAMP_MESSAGE.format(timestamp=timestamp) + image_prompt_string
                    image_prompt_strings += image_prompt_string

                image_prompt_strings += DEFAULT_MEDIA_OUTTRO
                sample = sample.replace(self.video_token, image_prompt_strings, 1)
            prompt_strings.append(sample)
        return prompt_strings

    @auto_docstring
    def __call__(
        self,
        images: ImageInput | list[ImageInput] | list[list[ImageInput]] = None,
        text: Union[TextInput, "PreTokenizedInput", list[TextInput], list["PreTokenizedInput"]] = None,
        videos: VideoInput | None = None,
        **kwargs: Unpack[SmolVLMProcessorKwargs],
    ) -> BatchEncoding:
        if text is None and images is None and videos is None:
            raise ValueError("You must provide one of `text`, `images` or `videos'.")

        if text is None and ((images is None) ^ (videos is not None)):
            raise ValueError("You must specify exactly one of `images` or `videos`")

        output_kwargs = self._merge_kwargs(
            SmolVLMProcessorKwargs,
            tokenizer_init_kwargs=self.tokenizer.init_kwargs,
            **kwargs,
        )

        if text is not None:
            if isinstance(text, str):
                text = [text]
            elif not isinstance(text, list) and not isinstance(text[0], str):
                raise ValueError("Invalid input text. Please provide a string, or a list of strings")
            n_images_in_text = sum(sample.count(self.image_token) for sample in text)
            if n_images_in_text > 0 and (images is None and videos is None):
                raise ValueError(f"We detected {n_images_in_text} tokens in the text but no images/videos were passed")

        inputs = {}
        # Images and videos are mutually exclusive, so process one which is present
        if images is not None:
            images = self.image_processor.fetch_images(images)
            images = make_nested_list_of_images(images)
            vision_inputs = self.image_processor(images, **output_kwargs["images_kwargs"])

            image_rows = vision_inputs.pop("rows", None)
            image_cols = vision_inputs.pop("cols", None)
            inputs.update(vision_inputs)

            if text is not None:
                n_images_in_text = [sample.count(self.image_token) for sample in text]
                n_images_in_images = [len(sublist) for sublist in images]
                if n_images_in_images != n_images_in_text:
                    raise ValueError(
                        f"The number of images in the text {n_images_in_text} and images {n_images_in_images} should be the same."
                    )
                # Set default values for image_rows and image_cols if not provided
                if image_rows is None:
                    image_rows = [[0] * n_images for n_images in n_images_in_text]
                if image_cols is None:
                    image_cols = [[0] * n_images for n_images in n_images_in_text]
                text = self.expand_text_with_image_tokens(text, image_rows=image_rows, image_cols=image_cols)

        elif videos is not None:
            vision_inputs = self.video_processor(videos, **output_kwargs["videos_kwargs"])
            if text is not None:
                n_videos_in_text = [sample.count(self.video_token) for sample in text]
                n_videos_in_videos = [len(sublist) for sublist in videos]
                if n_videos_in_videos != n_videos_in_text:
                    raise ValueError(
                        f"The number of videos in the text {n_videos_in_text} and videos {n_videos_in_videos} should be the same."
                    )
                text = self.expand_text_with_video_tokens(text, vision_inputs)

            # If user has not requested video metadata, pop it. By default metadata
            # is always returned to expand video tokens correctly
            if not kwargs.get("return_metadata"):
                vision_inputs.pop("video_metadata")
            inputs.update(vision_inputs)

        return_tensors = output_kwargs["text_kwargs"].pop("return_tensors", None)

        if text is not None:
            text_inputs = self.tokenizer(text, **output_kwargs["text_kwargs"])
            if hasattr(self, "_check_special_mm_tokens"):
                self._check_special_mm_tokens(text, text_inputs, modalities=["image"])
            inputs.update(text_inputs)

        return BatchFeature(inputs, tensor_type=return_tensors)

    def apply_chat_template(
        self,
        conversation: list[dict[str, str]] | list[list[dict[str, str]]],
        chat_template: str | None = None,
        processor_kwargs: dict | None = None,
        **kwargs,
    ) -> str:
        """
        Similar to the `apply_chat_template` method on tokenizers, this method applies a Jinja template to input
        conversations to turn them into a single tokenizable string.

        The input is expected to be in the following format, where each message content is a list consisting of text and
        optionally image or video inputs. One can also provide an image, video, URL or local path which will be used to form
        `pixel_values` when `return_dict=True`. If not provided, one will get only the formatted text, optionally tokenized text.

        conversation = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "url": "https://www.ilankelman.org/stopsigns/australia.jpg"},
                    {"type": "text", "text": "Please describe this image in detail."},
                ],
            },
        ]

        Args:
            conversation (`Union[list[Dict, [str, str]], list[list[dict[str, str]]]]`):
                The conversation to format.
            chat_template (`Optional[str]`, *optional*):
                The Jinja template to use for formatting the conversation. If not provided, the tokenizer's
                chat template is used.
        """
        if isinstance(conversation, (list, tuple)) and (
            isinstance(conversation[0], (list, tuple)) or hasattr(conversation[0], "content")
        ):
            conversations = conversation
        else:
            conversations = [conversation]

        has_video = any(
            (isinstance(content, dict) and content["type"] == "video")
            for conversation in conversations
            for message in conversation
            for content in message["content"]
        )
        if chat_template is None and has_video:
            # re-assign to the correct default template for BC, if user is not requesting their own template
            chat_template = DEFAULT_CHAT_TEMPLATE

        # Users might be passing processor kwargs simply as `**kwargs`
        # Guard against video_processor being None (image-only usage)
        if self.video_processor is not None:
            if processor_kwargs:
                processor_kwargs.setdefault("num_frames", self.video_processor.num_frames)
                processor_kwargs.setdefault("fps", self.video_processor.fps)
            else:
                kwargs.setdefault("num_frames", self.video_processor.num_frames)
                kwargs.setdefault("fps", self.video_processor.fps)

        return super().apply_chat_template(conversation, chat_template, processor_kwargs=processor_kwargs, **kwargs)


def run_foveated_tiling_tests():
    """Model-free tests for the foveated-tiling geometry (Stage 1).

    Validates ``gaze_to_tiles`` only (no torch / no model load), so it can run
    anywhere. The text-builder / invariant (Stage 2) is checked end-to-end in
    ``build_foveated_inputs`` via the ``assert`` on the ``<image>`` count.
    """
    print("\n" + "=" * 60)
    print("Running foveated-tiling geometry tests")
    print("=" * 60)

    R, C = 4, 3
    n_img = R * C + 1          # 13; global index = 12

    def chk(name, got, exp):
        assert got == exp, f"{name} FAIL: got {got}, expected {exp}"
        print(f"  {name} PASS  {got}")

    # interior gaze -> single tile + global
    chk("T1 interior",    gaze_to_tiles((0.5, 0.6), (1080, 1920), R, C, n_img), [7, 12])
    # gaze on a horizontal boundary (y=0.5) -> straddle adds the top neighbor
    chk("T2 boundary",    gaze_to_tiles((0.5, 0.5), (1080, 1920), R, C, n_img), [4, 7, 12])
    # top-left corner -> tile 0 + global (neighbors clamped away)
    chk("T3 corner",      gaze_to_tiles((0.0, 0.0), (1000, 1000), R, C, n_img), [0, 12])
    # near a vertical boundary (interior y) -> gaze tile + left neighbor + global
    chk("T4 left-neigh",  gaze_to_tiles((0.34, 0.6), (1000, 1000), R, C, n_img), [6, 7, 12])
    # pixel-coordinate gaze normalizes to the same as its fraction
    chk("T5 pixels",      gaze_to_tiles((540, 1152), (1080, 1920), R, C, n_img),
                          gaze_to_tiles((0.5, 0.6), (1080, 1920), R, C, n_img))
    # off-screen / negative -> global-only fallback
    chk("T6 offscreen",   gaze_to_tiles((2000, 960), (1080, 1920), R, C, n_img), [12])
    chk("T7 negative",    gaze_to_tiles((-0.1, 0.5), (1080, 1920), R, C, n_img), [12])
    # neighbors disabled -> exactly gaze tile + global
    chk("T8 no-neighbors", gaze_to_tiles((0.34, 0.6), (1000, 1000), R, C, n_img, neighbors=False), [7, 12])
    # bottom-right -> last tile + global
    chk("T9 bottom-right", gaze_to_tiles((0.99, 0.99), (1000, 1000), R, C, n_img), [11, 12])

    print("\nAll foveated-tiling geometry tests PASSED")
    print("=" * 60)


__all__ = ["SmolVLMProcessor", "gaze_to_tiles", "run_foveated_tiling_tests"]
