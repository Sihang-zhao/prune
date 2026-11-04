"""Native Qwen3-VL input preparation and unpruned inference."""

from __future__ import annotations
import time
from dataclasses import dataclass
from typing import Any, Mapping

try:
    import torch
except ImportError:
    torch = None


@dataclass
class ModelOutput:
    answer: str
    seconds: float
    input_tokens: int
    output_tokens: int
    image_tokens: int
    flops: int
    original_image_tokens: int = 0
    pruned_image_tokens: int = 0
    pruning_seconds: float = 0.0
    generation_seconds: float = 0.0
    decode_seconds: float = 0.0


def _dtype(name: str) -> torch.dtype | str:
    if torch is None:
        raise RuntimeError("PyTorch is required for Qwen inference: pip install torch")
    value = str(name or "auto").lower()
    aliases = {
        "fp16": torch.float16,
        "float16": torch.float16,
        "bf16": torch.bfloat16,
        "bfloat16": torch.bfloat16,
        "fp32": torch.float32,
        "float32": torch.float32,
        "auto": "auto",
    }
    if value not in aliases:
        raise ValueError(f"unsupported dtype {name!r}; expected one of auto, fp16, bf16, or fp32")
    return aliases[value]


def _get(mapping: Any, key: str, default: Any = 0) -> Any:
    if mapping is None:
        return default
    if isinstance(mapping, Mapping):
        return mapping.get(key, default)
    return getattr(mapping, key, default)


def _move_to_device(value: Any, device: Any) -> Any:
    """Move processor mappings on old Transformers versions without BatchFeature.to."""
    if torch is not None and torch.is_tensor(value):
        return value.to(device)
    if isinstance(value, Mapping):
        return {key: _move_to_device(item, device) for key, item in value.items()}
    if isinstance(value, tuple):
        return tuple((_move_to_device(item, device) for item in value))
    if isinstance(value, list):
        return [_move_to_device(item, device) for item in value]
    move = getattr(value, "to", None)
    if callable(move):
        try:
            return move(device)
        except (AttributeError, TypeError):
            pass
    return value


def estimate_flops(
    config: Any,
    input_tokens: int,
    output_tokens: int,
    image_tokens: int,
    *,
    vision_tokens: int | None = None,
    vision_prune_layer: int | None = None,
) -> int:
    """Estimate inference FLOPs from the model config."""
    text = _get(config, "text_config", config)
    layers = int(_get(text, "num_hidden_layers", 0) or 0)
    hidden = int(_get(text, "hidden_size", 0) or 0)
    intermediate = int(_get(text, "intermediate_size", 0) or 0)
    seq = max(1, int(input_tokens))
    new = max(0, int(output_tokens))
    per_token = 2 * (4 * hidden * hidden + 2 * hidden * intermediate)
    attention = 4 * hidden * seq
    text_flops = layers * (seq * (per_token + 2 * hidden * seq) + new * (per_token + attention))
    vision = _get(config, "vision_config", None)
    v_layers = int(_get(vision, "depth", _get(vision, "num_hidden_layers", 0)) or 0)
    v_hidden = int(_get(vision, "hidden_size", 0) or 0)
    v_intermediate = int(_get(vision, "intermediate_size", 0) or 0)
    patches = max(0, int(image_tokens if vision_tokens is None else vision_tokens))
    vision_token_layers = v_layers * patches
    if vision_prune_layer is not None:
        cut = int(vision_prune_layer)
        if not 0 < cut < v_layers:
            raise ValueError("vision_prune_layer must be strictly inside the visual depth")
        vision_token_layers = cut * patches + (v_layers - cut) * max(0, int(image_tokens))
    vision_flops = (
        vision_token_layers * 2 * (4 * v_hidden * v_hidden + 2 * v_hidden * v_intermediate)
    )
    return int(max(0, text_flops + vision_flops))


def _inference_mode():
    if torch is None:

        def decorator(function):
            return function

        return decorator
    return torch.inference_mode()


class QwenVL:
    """Qwen3-VL runner exposing ViT -> projection -> LLM -> answer stages."""

    SYSTEM_INSTRUCTION = "You are a careful visual question answering assistant. Inspect the provided image and answer the user's question. Follow the requested answer format and do not include analysis or extra explanation."

    def __init__(
        self,
        name: str = "Qwen/Qwen3-VL-8B-Instruct",
        device: str = "cuda:0",
        dtype: str = "auto",
        max_new_tokens: int = 32,
        trust_remote_code: bool = True,
        local_files_only: bool = False,
        max_image_pixels: int = 4194304,
        enable_thinking: bool = False,
    ) -> None:
        if torch is None:
            raise RuntimeError("PyTorch is required for Qwen inference: pip install torch")
        try:
            from transformers import AutoProcessor
        except ImportError as exc:
            raise RuntimeError("transformers is required: pip install transformers") from exc
        self.device = torch.device(device)
        self.max_new_tokens = int(max_new_tokens)
        if self.max_new_tokens <= 0:
            raise ValueError("max_new_tokens must be positive")
        self.max_image_pixels = int(max_image_pixels)
        if self.max_image_pixels <= 0:
            raise ValueError("max_image_pixels must be positive")
        self.enable_thinking = bool(enable_thinking)
        self.visual_pruning = False
        dtype_value = _dtype(dtype)
        if dtype_value == "auto":
            if self.device.type == "cuda":
                try:
                    bf16_ok = bool(torch.cuda.is_bf16_supported())
                except (AttributeError, RuntimeError):
                    bf16_ok = False
                dtype_value = torch.bfloat16 if bf16_ok else torch.float16
            else:
                dtype_value = torch.float32
        try:
            from transformers import Qwen3VLForConditionalGeneration
        except ImportError as exc:
            raise RuntimeError("This Transformers version does not support Qwen3-VL") from exc
        model_cls = Qwen3VLForConditionalGeneration
        kwargs: dict[str, Any] = {
            "trust_remote_code": trust_remote_code,
            "low_cpu_mem_usage": True,
            "local_files_only": local_files_only,
        }
        kwargs["dtype"] = dtype_value
        try:
            self.model = model_cls.from_pretrained(name, **kwargs)
        except TypeError:
            if "dtype" in kwargs:
                kwargs["torch_dtype"] = kwargs.pop("dtype")
            self.model = model_cls.from_pretrained(name, **kwargs)
        self.model.to(self.device)
        self.model.eval()
        generation_config = getattr(self.model, "generation_config", None)
        if generation_config is not None:
            for field_name in ("temperature", "top_p", "top_k"):
                if hasattr(generation_config, field_name):
                    setattr(generation_config, field_name, None)
        self.processor = AutoProcessor.from_pretrained(
            name, trust_remote_code=trust_remote_code, local_files_only=local_files_only
        )
        self.vit = getattr(self.model, "visual", getattr(self.model, "vision_tower", None))
        self.projection = getattr(self.vit, "merger", getattr(self.vit, "proj", None))
        self.llm = getattr(self.model, "language_model", self.model)

    @property
    def config(self) -> Any:
        return self.model.config

    def stages(self) -> tuple[str, ...]:
        stages = ["ViT", "Projection"]
        if self.visual_pruning:
            stages.append("VisualTokenPruning")
        stages.extend(("LLM", "Answer"))
        return tuple(stages)

    def _inputs(self, image: Any, prompt: str) -> Mapping[str, Any]:
        images = image if isinstance(image, (list, tuple)) else [image]
        content = [{"type": "image", "image": item} for item in images]
        content.append({"type": "text", "text": prompt})
        messages = [
            {"role": "system", "content": [{"type": "text", "text": self.SYSTEM_INSTRUCTION}]},
            {"role": "user", "content": content},
        ]
        try:
            rendered = self.processor.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
                enable_thinking=self.enable_thinking,
            )
        except TypeError:
            rendered = self.processor.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
        if isinstance(rendered, (tuple, list)):
            rendered = next((item for item in rendered if isinstance(item, str)), None)
        inputs = None
        if isinstance(rendered, str):
            per_image_pixels = max(65536, self.max_image_pixels // max(1, len(images)))
            kwargs = {
                "text": [rendered],
                "images": images,
                "padding": True,
                "return_tensors": "pt",
                "size": {
                    "longest_edge": per_image_pixels,
                    "shortest_edge": min(per_image_pixels, 65536),
                },
                "max_pixels": per_image_pixels,
            }
            try:
                inputs = self.processor(**kwargs)
            except TypeError:
                kwargs.pop("max_pixels", None)
                inputs = self.processor(**kwargs)
        if inputs is None:
            inputs = self.processor.apply_chat_template(
                messages,
                tokenize=True,
                add_generation_prompt=True,
                return_dict=True,
                return_tensors="pt",
            )
        return _move_to_device(inputs, self.device)

    def _compose_inputs_embeds(
        self, input_ids: torch.Tensor, image_features: torch.Tensor
    ) -> torch.Tensor:
        """Inject pruned image features into the compact image placeholders."""
        image_token_id = int(_get(self.config, "image_token_id", 151655))
        inputs_embeds = self.model.get_input_embeddings()(input_ids)
        image_mask = input_ids == image_token_id
        expected = int(image_mask.sum().item())
        if expected != int(image_features.shape[0]):
            raise RuntimeError(
                f"image placeholder count and pruned feature count differ: placeholders={expected}, features={image_features.shape[0]}"
            )
        inputs_embeds[image_mask] = image_features.to(
            device=inputs_embeds.device, dtype=inputs_embeds.dtype
        )
        return inputs_embeds

    def _pruned_inputs(self, inputs: Mapping[str, Any]) -> tuple[dict[str, Any], int, int]:
        """Return the uncompressed inputs; Q-FlowCover overrides this method."""
        image_tokens = self._image_tokens(inputs, self.config)
        return (dict(inputs), image_tokens, image_tokens)

    @staticmethod
    def _image_tokens(inputs: Mapping[str, Any], config: Any) -> int:
        grid = inputs.get("image_grid_thw")
        if grid is None:
            return 0
        values = grid.detach().reshape(-1).tolist()
        merge = int(_get(_get(config, "vision_config", None), "spatial_merge_size", 1) or 1)
        if len(values) < 3:
            return 0
        return int(
            sum(
                (
                    values[index] * values[index + 1] * values[index + 2]
                    for index in range(0, len(values), 3)
                )
            )
            // (merge * merge)
        )

    @_inference_mode()
    def answer(self, image: Any, prompt: str) -> ModelOutput:
        """Run ViT/projection/LLM and decode the answer for one case."""
        inputs = self._inputs(image, prompt)
        start = time.perf_counter()
        pruning_started = time.perf_counter()
        model_inputs, image_tokens, original_image_tokens = self._pruned_inputs(inputs)
        if self.device.type == "cuda":
            torch.cuda.synchronize(self.device)
        pruning_seconds = time.perf_counter() - pruning_started
        input_tokens = (
            int(model_inputs["input_ids"].shape[-1]) if "input_ids" in model_inputs else 0
        )
        generate_kwargs = {
            "max_new_tokens": self.max_new_tokens,
            "min_new_tokens": 1,
            "do_sample": False,
            "use_cache": True,
            "return_dict_in_generate": False,
            "output_scores": False,
            "output_attentions": False,
            "output_hidden_states": False,
        }
        generation_started = time.perf_counter()
        generated = self.model.generate(**model_inputs, **generate_kwargs)
        if self.device.type == "cuda":
            torch.cuda.synchronize(self.device)
        generation_seconds = time.perf_counter() - generation_started
        if hasattr(generated, "sequences"):
            generated = generated.sequences
        if torch is not None and torch.is_tensor(generated) and (generated.ndim == 1):
            generated = generated.unsqueeze(0)
        prompt_length = int(model_inputs["input_ids"].shape[-1])
        has_prompt_prefix = (
            torch is not None
            and torch.is_tensor(generated)
            and (generated.ndim == 2)
            and (int(generated.shape[-1]) >= prompt_length)
            and torch.equal(
                generated[:, :prompt_length],
                model_inputs["input_ids"].to(device=generated.device, dtype=generated.dtype),
            )
        )
        new_ids = generated[:, prompt_length:] if has_prompt_prefix else generated
        decode_started = time.perf_counter()
        text = self.processor.batch_decode(
            new_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False
        )[0].strip()
        if torch is not None and isinstance(new_ids, torch.Tensor):
            output_tokens = int(new_ids.shape[-1])
        else:
            output_tokens = int(new_ids[0].numel()) if new_ids else 0
        if not text:
            raise RuntimeError(
                f"the model returned no decodable answer tokens; continuation_tokens={output_tokens}"
            )
        decode_seconds = time.perf_counter() - decode_started
        seconds = time.perf_counter() - start
        return ModelOutput(
            answer=text,
            seconds=seconds,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            image_tokens=image_tokens,
            flops=estimate_flops(
                self.config,
                input_tokens,
                output_tokens,
                image_tokens,
                vision_tokens=original_image_tokens,
            ),
            original_image_tokens=original_image_tokens,
            pruned_image_tokens=image_tokens,
            pruning_seconds=pruning_seconds,
            generation_seconds=generation_seconds,
            decode_seconds=decode_seconds,
        )


class BaseQwenVL(QwenVL):
    """Native inference with the processor grid exposed for FLOPs accounting."""

    def _inputs(self, image: Any, prompt: str) -> Mapping[str, Any]:
        inputs = super()._inputs(image, prompt)
        grid = inputs.get("image_grid_thw")
        if grid is None:
            raise ValueError("Native baseline processor produced no image_grid_thw")
        self.last_image_grid_thw = grid.detach().reshape(-1, 3).cpu().tolist()
        return inputs


__all__ = ["ModelOutput", "QwenVL", "BaseQwenVL", "estimate_flops"]
