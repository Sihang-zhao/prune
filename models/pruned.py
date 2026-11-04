"""Complete-group prompt compaction and native Qwen3-VL feature replay."""

from __future__ import annotations
import time
import unicodedata
from contextlib import ExitStack, contextmanager
from dataclasses import dataclass, field
from typing import Any, Mapping, Sequence

try:
    import torch
    import torch.nn.functional as F
except ImportError:
    torch = None
    F = None
from .base import ModelOutput, QwenVL, _get, estimate_flops


def _is_early_middle_mode(mode: str) -> bool:
    return str(mode) == "middle_text_early"


def _inference_mode():
    if torch is None:

        def decorator(function):
            return function

        return decorator
    return torch.inference_mode()


@dataclass
class PrunedModelOutput(ModelOutput):
    """Baseline output plus metrics specific to the pruning experiment."""

    retained_tokens: int = 0
    retention_ratio: float = 0.0
    pruning_latency: float = 0.0
    prefill_latency: float = 0.0
    selection_budgets: list[tuple[int, int, int]] = field(default_factory=list)
    selected_indices: list[list[int]] = field(default_factory=list)
    selection_groups: list[dict[str, list[int]]] = field(default_factory=list)
    selection_mode: str = "middle_text_early"
    selection_strategy: str = "topk"
    selection_layer: int | None = None
    vision_layer_token_counts: list[int] = field(default_factory=list)
    vision_merge_unit: int = 0


@dataclass
class _SelectedVisionOutput:
    """Minimal encoder-output contract consumed by Qwen3-VL's multimodal path."""

    pooler_output: tuple[torch.Tensor, ...]
    deepstack_features: list[tuple[torch.Tensor, ...]]

    def __iter__(self):
        yield self.pooler_output
        yield self.deepstack_features

    def __getitem__(self, index: int):
        return (self.pooler_output, self.deepstack_features)[index]

    def __len__(self) -> int:
        return 2


class _FeatureTuple(tuple):
    """Tuple of per-view features compatible with tensor-only older models."""

    def _cat(self) -> torch.Tensor:
        if not self:
            raise RuntimeError("cannot materialize an empty visual feature tuple")
        return torch.cat(tuple(self), dim=0)

    def to(self, *args: Any, **kwargs: Any) -> torch.Tensor:
        return self._cat().to(*args, **kwargs)

    @property
    def shape(self) -> torch.Size:
        return self._cat().shape

    @property
    def device(self) -> torch.device:
        return self._cat().device

    @property
    def dtype(self) -> torch.dtype:
        return self._cat().dtype


@contextmanager
def _temporary_method(module: Any, name: str, replacement: Any):
    """Temporarily replace a model method and restore its exact state."""
    had_value = name in getattr(module, "__dict__", {})
    old_value = getattr(module, "__dict__", {}).get(name)
    setattr(module, name, replacement)
    try:
        yield
    finally:
        if had_value:
            setattr(module, name, old_value)
        else:
            try:
                delattr(module, name)
            except AttributeError:
                pass


def _require_torch() -> None:
    if torch is None or F is None:
        raise RuntimeError("PyTorch is required for training-free visual pruning")


def _normalize_model_answer(text: str) -> str:
    """Strip Qwen/RS-OPSD thinking and answer wrappers before scoring."""
    answer = str(text or "").strip()
    think_end = answer.rfind("</think>")
    if think_end != -1:
        answer = answer[think_end + len("</think>") :].strip()
    start = answer.rfind("<answer>")
    end = answer.find("</answer>", start + len("<answer>")) if start != -1 else -1
    if start != -1 and end > start:
        return answer[start + len("<answer>") : end].strip()
    lowered = answer.casefold()
    for marker in ("the answer is:", "answer:"):
        marker_start = lowered.rfind(marker)
        if marker_start != -1:
            return answer[marker_start + len(marker) :].strip()
    return answer


def _move_processor_value(value: Any, device: Any) -> Any:
    """Move a processor result without assuming it is a BatchFeature."""
    if isinstance(value, Mapping):
        return {key: _move_processor_value(item, device) for key, item in value.items()}
    if isinstance(value, tuple):
        return tuple((_move_processor_value(item, device) for item in value))
    if isinstance(value, list):
        return [_move_processor_value(item, device) for item in value]
    move = getattr(value, "to", None)
    if callable(move):
        try:
            return move(device)
        except (AttributeError, TypeError):
            try:
                return move(device=device)
            except (AttributeError, TypeError):
                pass
    return value


def _find_processor_inputs(value: Any, device: Any) -> Any | None:
    """Find the dict-like processor payload inside version-specific wrappers."""
    if isinstance(value, (str, bytes, bytearray)):
        return None
    if isinstance(value, Mapping):
        if "input_ids" in value:
            return _move_processor_value(value, device)
        for item in value.values():
            found = _find_processor_inputs(item, device)
            if found is not None:
                return found
        return None
    try:
        has_input_ids = getattr(value, "input_ids", None) is not None
    except Exception:
        has_input_ids = False
    if has_input_ids:
        return _move_processor_value(value, device)
    if isinstance(value, (tuple, list)):
        for item in value:
            found = _find_processor_inputs(item, device)
            if found is not None:
                return found
    return None


def _as_mapping(value: Any) -> Mapping[str, Any] | None:
    """Return a mapping view for BatchFeature-like objects."""
    if isinstance(value, Mapping):
        return value
    try:
        keys = value.keys()
        return {key: value[key] for key in keys}
    except (AttributeError, KeyError, TypeError):
        return None


def _tensor_leaves(value: Any) -> list[torch.Tensor] | None:
    """Collect tensors from a tuple/list wrapper without guessing field names."""
    if torch is None:
        return None
    if torch.is_tensor(value):
        return [value]
    if isinstance(value, (tuple, list)):
        leaves: list[torch.Tensor] = []
        for item in value:
            nested = _tensor_leaves(item)
            if nested is None:
                return None
            leaves.extend(nested)
        return leaves
    return None


def _normalize_tensor_field(name: str, value: Any, device: Any) -> Any:
    """Normalize version-specific processor fields to the model's tensor API."""
    if torch is None:
        return value
    if torch.is_tensor(value):
        return value.to(device)
    leaves = _tensor_leaves(value)
    if not leaves:
        return value
    tensors = [item.to(device) for item in leaves]
    if len(tensors) == 1:
        return tensors[0]
    try:
        if name in {"image_grid_thw", "video_grid_thw"}:
            if any((item.numel() % 3 != 0 for item in tensors)):
                raise ValueError("grid tensor does not contain triples")
            return torch.cat([item.reshape(-1, 3) for item in tensors], dim=0)
        if name == "pixel_values" or name == "pixel_values_videos":
            ranks = {int(item.ndim) for item in tensors}
            if len(ranks) != 1 or any(
                (tuple(item.shape[1:]) != tuple(tensors[0].shape[1:]) for item in tensors)
            ):
                raise ValueError("pixel tensors have incompatible shapes")
            return torch.cat(tensors, dim=0)
        if name in {"input_ids", "attention_mask", "mm_token_type_ids", "token_type_ids"}:
            if all((item.ndim == 1 for item in tensors)):
                lengths = {int(item.shape[0]) for item in tensors}
                if len(lengths) != 1:
                    raise ValueError("sequence tensors have different lengths")
                return torch.stack(tensors, dim=0)
            if all((item.ndim == 2 for item in tensors)):
                if any((tuple(item.shape[1:]) != tuple(tensors[0].shape[1:]) for item in tensors)):
                    raise ValueError("batched sequence tensors have incompatible shapes")
                return torch.cat(tensors, dim=0)
        if name == "position_ids":
            if all((item.ndim == 3 for item in tensors)):
                if any(
                    (
                        item.shape[0] != tensors[0].shape[0] or item.shape[2] != tensors[0].shape[2]
                        for item in tensors
                    )
                ):
                    raise ValueError("position tensors have incompatible modal/sequence dimensions")
                return torch.cat(tensors, dim=1)
    except (RuntimeError, ValueError) as exc:
        raise RuntimeError(
            f"processor field {name!r} contains multiple tensors that cannot be normalized; shapes={[tuple(item.shape) for item in tensors]}"
        ) from exc
    raise RuntimeError(
        f"processor field {name!r} was returned as a nested tensor sequence; unsupported shapes={[tuple(item.shape) for item in tensors]}"
    )


def _normalize_processor_mapping(mapping: Mapping[str, Any], device: Any) -> dict[str, Any]:
    """Move and normalize all tensor-valued multimodal processor fields."""
    tensor_fields = {
        "input_ids",
        "attention_mask",
        "token_type_ids",
        "mm_token_type_ids",
        "position_ids",
        "pixel_values",
        "pixel_values_videos",
        "image_grid_thw",
        "video_grid_thw",
    }
    result: dict[str, Any] = {}
    for key, value in mapping.items():
        if key in tensor_fields:
            result[key] = _normalize_tensor_field(key, value, device)
        else:
            result[key] = _move_processor_value(value, device)
    return result


def _infer_tensor_processor_inputs(value: Any, device: Any) -> Mapping[str, Any] | None:
    """Recover named multimodal fields from a legacy tuple of tensors."""
    if torch is None or not isinstance(value, (tuple, list)):
        return None
    tensors = [item for item in value if torch.is_tensor(item)]
    if len(tensors) != len(value):
        return None
    grid_candidates = [item for item in tensors if item.ndim >= 1 and int(item.shape[-1]) == 3]
    pixel_candidates = [
        item for item in tensors if torch.is_floating_point(item) and item.ndim >= 2
    ]
    integer_2d = [item for item in tensors if not torch.is_floating_point(item) and item.ndim == 2]
    if len(grid_candidates) != 1 or len(pixel_candidates) != 1 or (not integer_2d):
        return None
    grid = grid_candidates[0]
    pixels = pixel_candidates[0]
    integer_2d = [item for item in integer_2d if item is not grid]
    if not integer_2d:
        return None
    ids = max(integer_2d, key=lambda item: int(item.detach().max().item()) if item.numel() else -1)
    result: dict[str, Any] = {
        "input_ids": ids.to(device),
        "pixel_values": pixels.to(device),
        "image_grid_thw": grid.to(device),
    }
    for item in integer_2d:
        if item is ids or tuple(item.shape) != tuple(ids.shape):
            continue
        maximum = int(item.detach().max().item()) if item.numel() else 0
        minimum = int(item.detach().min().item()) if item.numel() else 0
        if minimum >= 0 and maximum <= 1:
            result.setdefault("attention_mask", item.to(device))
        elif minimum >= 0 and maximum <= 2:
            result.setdefault("mm_token_type_ids", item.to(device))
    return result


def _is_punctuation_only(decoded: str) -> bool:
    stripped = "".join((char for char in str(decoded) if not char.isspace()))
    stripped = stripped.lstrip("▁Ġ")
    if stripped.startswith("##"):
        stripped = stripped[2:]
    if not stripped:
        return True
    if stripped.startswith("<|") or (stripped.startswith("<") and stripped.endswith(">")):
        return True
    return all((unicodedata.category(char).startswith("P") for char in stripped))


def _decode_token(tokenizer: Any, token_id: int) -> str:
    if tokenizer is None:
        return str(token_id)
    decode = getattr(tokenizer, "decode", None)
    if not callable(decode):
        raise RuntimeError("tokenizer must expose decode() for text-token filtering")
    try:
        return str(
            decode([int(token_id)], skip_special_tokens=False, clean_up_tokenization_spaces=False)
        )
    except TypeError:
        return str(decode([int(token_id)], skip_special_tokens=False))


def valid_text_token_mask(
    input_ids: torch.Tensor,
    tokenizer: Any = None,
    attention_mask: torch.Tensor | None = None,
    *,
    pad_token_id: int | None = None,
    special_token_ids: Sequence[int] = (),
    extra_special_token_ids: Sequence[int] = (),
) -> torch.Tensor:
    """Return a mask excluding padding, special tokens, and pure punctuation."""
    _require_torch()
    if input_ids.ndim not in (1, 2):
        raise ValueError("input_ids must have shape [sequence] or [batch, sequence]")
    ids = input_ids if input_ids.ndim == 2 else input_ids.unsqueeze(0)
    if attention_mask is None:
        active = torch.ones(ids.shape, dtype=torch.bool, device=ids.device)
    else:
        mask = attention_mask if attention_mask.ndim == 2 else attention_mask.unsqueeze(0)
        if mask.shape != ids.shape:
            raise ValueError("attention_mask and input_ids must have matching shapes")
        active = mask.to(dtype=torch.bool)
    special = {int(value) for value in special_token_ids}
    special.update((int(value) for value in extra_special_token_ids))
    if pad_token_id is not None:
        special.add(int(pad_token_id))
    if tokenizer is not None:
        special.update((int(value) for value in getattr(tokenizer, "all_special_ids", ()) or ()))
    result = torch.zeros(ids.shape, dtype=torch.bool, device=ids.device)
    for row in range(ids.shape[0]):
        for col in range(ids.shape[1]):
            token_id = int(ids[row, col].item())
            if not bool(active[row, col]) or token_id in special:
                continue
            decoded = _decode_token(tokenizer, token_id)
            if not _is_punctuation_only(decoded):
                result[row, col] = True
    return result[0] if input_ids.ndim == 1 else result


def _first_tensor(value: Any) -> torch.Tensor | None:
    if torch is not None and torch.is_tensor(value):
        return value
    for key in ("last_hidden_state", "hidden_states", "pooler_output"):
        candidate = getattr(value, key, None)
        if candidate is not None and candidate is not value:
            found = _first_tensor(candidate)
            if found is not None:
                return found
    if isinstance(value, Mapping):
        for key in ("last_hidden_state", "image_embeds", "image_features", "hidden_states"):
            if key in value:
                found = _first_tensor(value[key])
                if found is not None:
                    return found
    if isinstance(value, (tuple, list)):
        for item in value:
            found = _first_tensor(item)
            if found is not None:
                return found
    return None


class PrunedQwenVL(QwenVL):
    """Multimodal plumbing for one early, complete-group pruning decision."""

    SYSTEM_INSTRUCTION = QwenVL.SYSTEM_INSTRUCTION

    def __init__(
        self,
        name="Qwen/Qwen3-VL-8B-Instruct",
        device="cuda:0",
        dtype="auto",
        max_new_tokens=32,
        trust_remote_code=True,
        local_files_only=False,
        visual_pruning=True,
        visual_keep_ratio=0.5,
        visual_keep_tokens=None,
        vision_layers=(6, 18, 27),
        pruning_mode="middle_text_early",
        pruning_layer=None,
        pruning_strategy="topk",
        max_image_pixels=4194304,
        enable_thinking=False,
    ):
        _require_torch()
        if not visual_pruning or pruning_mode != "middle_text_early" or pruning_strategy != "topk":
            raise ValueError("perturbation pruning requires one middle_text_early/topk decision")
        if not 0 < float(visual_keep_ratio) <= 1:
            raise ValueError("visual_keep_ratio must be in (0,1]")
        if visual_keep_tokens is not None and int(visual_keep_tokens) <= 0:
            raise ValueError("visual_keep_tokens must be positive")
        super().__init__(
            name=name,
            device=device,
            dtype=dtype,
            max_new_tokens=max_new_tokens,
            trust_remote_code=trust_remote_code,
            local_files_only=local_files_only,
            max_image_pixels=max_image_pixels,
            enable_thinking=enable_thinking,
        )
        self.vision_layers = tuple((int(n) for n in vision_layers))
        self.pruning_layer = int(
            pruning_layer if pruning_layer is not None else self.vision_layers[1]
        )
        self.pruning_mode = "middle_text_early"
        self.pruning_strategy = "topk"
        self.visual_keep_ratio = float(visual_keep_ratio)
        self.visual_keep_tokens = (
            int(visual_keep_tokens) if visual_keep_tokens is not None else None
        )
        self.visual_pruning = True
        inner_model = getattr(self.model, "model", None)
        if self.vit is None and inner_model is not None:
            self.vit = getattr(inner_model, "visual", getattr(inner_model, "vision_tower", None))
        if self.projection is None and self.vit is not None:
            self.projection = getattr(self.vit, "merger", getattr(self.vit, "proj", None))
        self.llm = getattr(inner_model, "language_model", self.llm)
        self._last_selection = {}
        self._last_rope_deltas = None
        self._last_compact_rope_deltas = None
        self._last_native_deepstack = []
        self._last_native_final_features = []
        self._last_early_selections = []
        self._last_early_original_counts = []
        self._last_vision_layer_token_counts = []
        self._last_cached_visual = None
        self._freeze_projection()

    def _inputs(self, image: Any, prompt: str) -> Mapping[str, Any]:
        """Build the LLM prompt with a system instruction and user image/question."""
        images = image if isinstance(image, (list, tuple)) else [image]
        content = [{"type": "image", "image": item} for item in images]
        content.append({"type": "text", "text": prompt})
        messages = [
            {"role": "system", "content": [{"type": "text", "text": self.SYSTEM_INSTRUCTION}]},
            {"role": "user", "content": content},
        ]
        inputs = None
        normalized = None
        render = getattr(self.processor, "apply_chat_template", None)
        process = self.processor if callable(self.processor) else None
        rendered = None
        if callable(render):
            try:
                rendered = render(
                    messages,
                    tokenize=False,
                    add_generation_prompt=True,
                    enable_thinking=self.enable_thinking,
                )
            except TypeError:
                rendered = render(messages, tokenize=False, add_generation_prompt=True)
            if isinstance(rendered, (tuple, list)):
                rendered = next((item for item in rendered if isinstance(item, str)), None)
        if isinstance(rendered, str) and process is not None:
            per_image_pixels = max(65536, self.max_image_pixels // max(1, len(images)))
            try:
                inputs = process(
                    text=[rendered],
                    images=images,
                    padding=True,
                    return_tensors="pt",
                    size={
                        "longest_edge": per_image_pixels,
                        "shortest_edge": min(per_image_pixels, 65536),
                    },
                    max_pixels=per_image_pixels,
                    do_sample_frames=False,
                )
            except TypeError:
                try:
                    inputs = process(
                        text=[rendered],
                        images=images,
                        padding=True,
                        return_tensors="pt",
                        size={
                            "longest_edge": per_image_pixels,
                            "shortest_edge": min(per_image_pixels, 65536),
                        },
                        max_pixels=per_image_pixels,
                    )
                except TypeError:
                    try:
                        inputs = process(text=[rendered], images=images, return_tensors="pt")
                    except TypeError:
                        inputs = None
            normalized = _find_processor_inputs(inputs, self.device)
        if normalized is None and callable(render):
            inputs = render(
                messages,
                tokenize=True,
                add_generation_prompt=True,
                return_dict=True,
                return_tensors="pt",
            )
            normalized = _find_processor_inputs(inputs, self.device)
            if normalized is None:
                normalized = _infer_tensor_processor_inputs(inputs, self.device)
        if normalized is None:
            raise RuntimeError(
                f"processor.apply_chat_template returned no payload containing input_ids; type={type(inputs).__name__}, value_types={([type(item).__name__ for item in inputs] if isinstance(inputs, (tuple, list)) else 'n/a')}"
            )
        mapping = _as_mapping(normalized)
        if mapping is None:
            raise RuntimeError(
                f"processor payload containing input_ids is not mapping-like; type={type(normalized).__name__}"
            )
        try:
            mapping = _normalize_processor_mapping(mapping, self.device)
        except Exception as exc:
            raise RuntimeError(
                f"processor tensor normalization failed before ViT pruning: {type(exc).__name__}: {exc}"
            ) from exc
        mapping = {key: value for key, value in mapping.items() if key != "token_type_ids"}
        missing = [
            key
            for key in ("input_ids", "pixel_values", "image_grid_thw")
            if mapping.get(key) is None
        ]
        if missing:
            present = list(mapping.keys())
            raise RuntimeError(
                f"processor output is missing required multimodal fields {missing}; present={present}; prompt/image construction cannot continue"
            )
        for key in ("input_ids", "pixel_values", "image_grid_thw"):
            if torch is not None and (not torch.is_tensor(mapping[key])):
                raise RuntimeError(
                    f"processor field {key!r} is {type(mapping[key]).__name__}, expected a Tensor; the multimodal model cannot call .to() on this value"
                )
        if mapping["input_ids"].ndim != 2:
            raise RuntimeError(
                f"processor input_ids must have shape [batch, sequence], got {tuple(mapping['input_ids'].shape)}"
            )
        if mapping["image_grid_thw"].numel() == 0 or mapping["image_grid_thw"].numel() % 3:
            raise RuntimeError(
                "processor image_grid_thw must contain one (T,H,W) triple per image view"
            )
        return mapping

    def _freeze_projection(self) -> None:
        projection = self.projection
        if projection is None or not callable(projection):
            raise RuntimeError(
                "the loaded ViT exposes no callable visual projector/merger; intermediate hidden states cannot be mapped to text space"
            )
        parameters = getattr(projection, "parameters", None)
        if callable(parameters):
            for parameter in parameters():
                parameter.requires_grad_(False)

    def _project_layer(
        self, state: torch.Tensor, layer: int, image_grid_thw: torch.Tensor
    ) -> torch.Tensor:
        projection = self.projection
        try:
            projected = projection(state)
        except (TypeError, RuntimeError):
            try:
                projected = projection(state, image_grid_thw)
            except Exception as second_error:
                raise RuntimeError(
                    f"visual projector cannot map ViT layer {layer} to text space; refusing a hidden-state-distance fallback"
                ) from second_error
        tensor = _first_tensor(projected)
        if tensor is None:
            raise RuntimeError(f"visual projector returned no tensor for ViT layer {layer}")
        return tensor

    def _text_tokens(
        self, inputs: Mapping[str, Any], scoring_text: str | None = None
    ) -> torch.Tensor:
        tokenizer = getattr(self.processor, "tokenizer", self.processor)
        config = self.config
        image_id = int(_get(config, "image_token_id", -1))
        video_id = int(_get(config, "video_token_id", -1))
        pad_id = getattr(tokenizer, "pad_token_id", None)
        if scoring_text is not None:
            encode = getattr(tokenizer, "__call__", None)
            if not callable(encode):
                raise RuntimeError(
                    "the processor tokenizer cannot encode the question for visual scoring"
                )
            encoded = encode(scoring_text, add_special_tokens=False, return_tensors="pt")
            encoded_mapping = _as_mapping(encoded)
            if encoded_mapping is not None:
                raw_ids = encoded_mapping.get("input_ids")
                raw_attention = encoded_mapping.get("attention_mask")
            else:
                leaves = _tensor_leaves(encoded)
                if not leaves:
                    raise RuntimeError(
                        f"question tokenizer returned no tensor input_ids; type={type(encoded).__name__}"
                    )
                integer_leaves = [item for item in leaves if not torch.is_floating_point(item)]
                if not integer_leaves:
                    raise RuntimeError(
                        f"question tokenizer returned no integer input_ids; shapes={[tuple(item.shape) for item in leaves]}"
                    )
                raw_ids = max(integer_leaves, key=lambda item: int(item.numel()))
                raw_attention = next(
                    (
                        item
                        for item in integer_leaves
                        if item is not raw_ids
                        and tuple(item.shape) == tuple(raw_ids.shape)
                        and (item.numel() > 0)
                        and (int(item.min().item()) >= 0)
                        and (int(item.max().item()) <= 1)
                    ),
                    None,
                )
            if raw_ids is None:
                raise RuntimeError("question tokenizer output is missing input_ids")
            input_ids = _normalize_tensor_field("input_ids", raw_ids, self.device)
            attention_mask = (
                _normalize_tensor_field("attention_mask", raw_attention, self.device)
                if raw_attention is not None
                else None
            )
            if not torch.is_tensor(input_ids) or input_ids.ndim != 2:
                raise RuntimeError(
                    f"question tokenizer input_ids must be a rank-2 Tensor; got {type(input_ids).__name__}"
                )
        else:
            input_ids = inputs["input_ids"]
            if input_ids.shape[0] != 1:
                raise RuntimeError(
                    "training-free token scoring currently requires a single prompt batch"
                )
            attention_mask = inputs.get("attention_mask")
        mask = valid_text_token_mask(
            input_ids[0],
            tokenizer,
            attention_mask[0] if attention_mask is not None else None,
            pad_token_id=pad_id,
            extra_special_token_ids=(image_id, video_id),
        )
        ids = input_ids[0][mask]
        if ids.numel() == 0:
            raise RuntimeError("no valid question/option text tokens remain after filtering")
        embeddings = self.model.get_input_embeddings()(ids.unsqueeze(0)).squeeze(0)
        if embeddings.ndim != 2:
            embeddings = embeddings.reshape(-1, int(embeddings.shape[-1]))
        return embeddings

    @staticmethod
    def _grid_counts(image_grid_thw: torch.Tensor, merge_size: int) -> list[int]:
        values = image_grid_thw.detach().reshape(-1, 3)
        counts = (values.prod(dim=-1) // max(1, int(merge_size)) ** 2).tolist()
        counts = [int(value) for value in counts]
        if any((value <= 0 for value in counts)):
            raise RuntimeError(f"invalid image_grid_thw for visual pruning: {image_grid_thw}")
        return counts

    def _keep_count(self, total: int) -> int:
        """Compute the per-view budget from the requested retention ratio."""
        if self.visual_keep_tokens is not None:
            keep = self.visual_keep_tokens
        else:
            keep = int(round(float(total) * self.visual_keep_ratio))
        keep = max(1, int(keep))
        if keep > int(total):
            raise RuntimeError(
                f"requested visual retention exceeds available tokens: keep={keep}, available={total}, ratio={self.visual_keep_ratio:.4f}"
            )
        return keep

    @staticmethod
    def _replace_groups(
        row: torch.Tensor,
        counts: Sequence[int],
        replacements: Sequence[torch.Tensor],
        image_token_id: int,
        *,
        reference: torch.Tensor | None = None,
    ) -> torch.Tensor:
        source = reference if reference is not None else row
        positions = torch.where(source == int(image_token_id))[0]
        if int(positions.numel()) != sum((int(value) for value in counts)):
            raise RuntimeError(
                f"image placeholder count does not match projected visual tokens: placeholders={positions.numel()}, features={sum(counts)}"
            )
        chunks: list[torch.Tensor] = []
        cursor = 0
        last = 0
        for count, replacement in zip(counts, replacements):
            count = int(count)
            start = int(positions[cursor].item())
            end = int(positions[cursor + count - 1].item()) + 1
            chunks.extend((row[last:start], replacement))
            last = end
            cursor += count
        chunks.append(row[last:])
        return torch.cat(chunks, dim=0)

    @staticmethod
    def _compact_grid(
        image_grid_thw: torch.Tensor, keep_counts: Sequence[int], merge_size: int
    ) -> torch.Tensor:
        rows = [(1, int(merge_size), max(1, int(count)) * int(merge_size)) for count in keep_counts]
        return torch.tensor(rows, device=image_grid_thw.device, dtype=image_grid_thw.dtype)

    def _original_position_ids(
        self, inputs: Mapping[str, Any], counts: Sequence[int], merge_size: int
    ) -> torch.Tensor:
        self._last_rope_deltas = None
        for target in (self.model, getattr(self.model, "model", None)):
            if target is not None and hasattr(target, "rope_deltas"):
                target.rope_deltas = None
        if inputs.get("position_ids") is not None:
            return inputs["position_ids"]
        method_candidates = (
            getattr(self.model, "get_rope_index", None),
            getattr(self.model, "compute_3d_position_ids", None),
            getattr(getattr(self.model, "model", None), "get_rope_index", None),
            getattr(getattr(self.model, "model", None), "compute_3d_position_ids", None),
            getattr(self.llm, "get_rope_index", None),
            getattr(self.llm, "compute_3d_position_ids", None),
        )
        for method in method_candidates:
            if not callable(method):
                continue
            attempts = (
                {
                    "input_ids": inputs["input_ids"],
                    "mm_token_type_ids": inputs.get("mm_token_type_ids"),
                    "image_grid_thw": inputs["image_grid_thw"],
                    "video_grid_thw": inputs.get("video_grid_thw"),
                    "attention_mask": inputs.get("attention_mask"),
                },
                {
                    "input_ids": inputs["input_ids"],
                    "mm_token_type_ids": inputs.get("mm_token_type_ids"),
                    "image_grid_thw": inputs["image_grid_thw"],
                    "attention_mask": inputs.get("attention_mask"),
                },
            )
            if _is_early_middle_mode(self.pruning_mode):
                attempts = (
                    {
                        "input_ids": inputs["input_ids"],
                        "image_grid_thw": inputs["image_grid_thw"],
                        "video_grid_thw": inputs.get("video_grid_thw"),
                        "attention_mask": inputs.get("attention_mask"),
                    },
                ) + attempts
            for kwargs in attempts:
                try:
                    value = method(**kwargs)
                    if isinstance(value, (tuple, list)):
                        self._last_rope_deltas = value[1] if len(value) > 1 else None
                        return value[0]
                    self._last_rope_deltas = None
                    return value
                except Exception:
                    continue
        return self._fallback_position_ids(inputs, counts, merge_size)

    def _fallback_position_ids(
        self, inputs: Mapping[str, Any], counts: Sequence[int], merge_size: int
    ) -> torch.Tensor:
        ids = inputs["input_ids"]
        image_id = int(_get(self.config, "image_token_id", 151655))
        grid = inputs["image_grid_thw"].detach().reshape(-1, 3)
        output = torch.zeros((3, ids.shape[0], ids.shape[1]), dtype=torch.long, device=ids.device)
        for row in range(ids.shape[0]):
            positions = torch.where(ids[row] == image_id)[0]
            cursor = 0
            max_coord = 0
            for image_index, count in enumerate(counts):
                image_positions = positions[cursor : cursor + int(count)]
                t, h, w = (int(value) for value in grid[image_index].tolist())
                hm, wm = (h // max(1, merge_size), w // max(1, merge_size))
                for local, position in enumerate(image_positions.tolist()):
                    per_frame = max(1, hm * wm)
                    tt, rem = divmod(local, per_frame)
                    hh, ww = divmod(rem, max(1, wm))
                    output[:, row, position] = torch.tensor((tt, hh, ww), device=ids.device)
                    max_coord = max(max_coord, tt, hh, ww)
                cursor += int(count)
            text_position = max_coord + 1
            for position in range(ids.shape[1]):
                if int(ids[row, position]) != image_id:
                    output[:, row, position] = text_position
                    text_position += 1
        return output

    @staticmethod
    def _compact_positions(
        position_ids: torch.Tensor,
        input_ids: torch.Tensor,
        counts: Sequence[int],
        selected: Sequence[torch.Tensor],
        image_token_id: int,
    ) -> torch.Tensor:
        if position_ids.ndim == 2:
            position_ids = position_ids.unsqueeze(0).expand(3, -1, -1)
        elif (
            position_ids.ndim == 3
            and position_ids.shape[0] == input_ids.shape[0]
            and (position_ids.shape[1] in (3, 4))
        ):
            position_ids = position_ids.transpose(0, 1)
        if position_ids.ndim != 3 or position_ids.shape[1] != input_ids.shape[0]:
            raise RuntimeError(
                "position_ids must have shape [modal_dims, batch, sequence] to preserve original visual coordinates"
            )
        rows: list[torch.Tensor] = []
        for row in range(input_ids.shape[0]):
            source = input_ids[row]
            positions = torch.where(source == int(image_token_id))[0]
            chunks: list[torch.Tensor] = []
            cursor = 0
            last = 0
            for count, indices in zip(counts, selected):
                count = int(count)
                start = int(positions[cursor].item())
                end = int(positions[cursor + count - 1].item()) + 1
                chunks.append(position_ids[:, row, last:start])
                chunks.append(position_ids[:, row, positions[cursor : cursor + count][indices]])
                cursor += count
                last = end
            chunks.append(position_ids[:, row, last:])
            rows.append(torch.cat(chunks, dim=-1))
        return torch.stack(rows, dim=1)

    @staticmethod
    def _rope_delta(position_ids: torch.Tensor, input_ids: torch.Tensor) -> torch.Tensor:
        """Derive Qwen's compact-sequence mRoPE delta from position IDs."""
        if position_ids.ndim != 3:
            raise RuntimeError(
                f"compact position_ids must be rank 3, got {tuple(position_ids.shape)}"
            )
        batch = int(input_ids.shape[0])
        if position_ids.shape[0] == batch and position_ids.shape[1] <= 4:
            maximum = position_ids.amax(dim=(1, 2))
        elif position_ids.shape[1] == batch:
            maximum = position_ids.amax(dim=(0, 2))
        else:
            raise RuntimeError(
                f"unsupported compact position_ids layout: position_ids={tuple(position_ids.shape)} input_ids={tuple(input_ids.shape)}"
            )
        return (maximum + 1 - int(input_ids.shape[1])).to(input_ids.dtype).view(-1, 1)

    def _vision_forward(self, inputs, *, text_tokens):
        pixels, grid = (inputs.get("pixel_values"), inputs.get("image_grid_thw"))
        if not torch.is_tensor(pixels) or not torch.is_tensor(grid):
            raise RuntimeError("pixel_values and image_grid_thw must be tensors")
        visual_dtype = getattr(self.vit, "dtype", None)
        if visual_dtype is not None and torch.is_floating_point(pixels):
            pixels = pixels.to(dtype=visual_dtype)
        return self._vision_forward_middle_early(pixels, grid, text_tokens)

    def _pruned_inputs(self, inputs, *, scoring_text=None):
        for key in ("input_ids", "image_grid_thw"):
            if not torch.is_tensor(inputs.get(key)):
                raise RuntimeError(f"visual pruning requires tensor {key}")
        image_grid = inputs["image_grid_thw"]
        merge_size = int(
            _get(_get(self.config, "vision_config", None), "spatial_merge_size", 1) or 1
        )
        counts = self._grid_counts(image_grid, merge_size)
        text_tokens = self._text_tokens(inputs, scoring_text=scoring_text)
        self._vision_forward(inputs, text_tokens=text_tokens)
        selections = list(self._last_early_selections)
        selected_features = list(self._last_native_final_features)
        keep_counts = [int(selection.indices.numel()) for selection in selections]
        if len(selected_features) != len(counts) or len(selections) != len(counts):
            raise RuntimeError("the visual pass must return one compact stream per view")
        expected_dim = int(selected_features[0].shape[-1]) if selected_features else 0
        if expected_dim <= 0 or any(
            (
                int(piece.shape[0]) != keep or int(piece.shape[-1]) != expected_dim
                for piece, keep in zip(selected_features, keep_counts)
            )
        ):
            raise RuntimeError("final visual features do not match the selected groups")
        del text_tokens
        config_image_id = int(_get(self.config, "image_token_id", 151655))
        selected_replacements = [
            inputs["input_ids"].new_full((keep,), config_image_id) for keep in keep_counts
        ]
        compact_ids = torch.stack(
            [
                self._replace_groups(
                    inputs["input_ids"][row], counts, selected_replacements, config_image_id
                )
                for row in range(inputs["input_ids"].shape[0])
            ]
        )
        compact_attention = None
        if inputs.get("attention_mask") is not None:
            compact_attention = torch.stack(
                [
                    self._replace_groups(
                        inputs["attention_mask"][row],
                        counts,
                        [inputs["attention_mask"].new_ones((keep,)) for keep in keep_counts],
                        config_image_id,
                        reference=inputs["input_ids"][row],
                    )
                    for row in range(inputs["attention_mask"].shape[0])
                ]
            )
        compact_grid = self._compact_grid(image_grid, keep_counts, merge_size)
        model_inputs = dict(inputs)
        model_inputs["input_ids"] = compact_ids
        model_inputs["image_grid_thw"] = compact_grid
        model_inputs.pop("inputs_embeds", None)
        if compact_attention is not None:
            model_inputs["attention_mask"] = compact_attention
        original_positions = self._original_position_ids(inputs, counts, merge_size)
        compact_positions = self._compact_positions(
            original_positions,
            inputs["input_ids"],
            counts,
            [selection.indices for selection in selections],
            config_image_id,
        )
        model_inputs["position_ids"] = compact_positions
        self._last_compact_rope_deltas = self._rope_delta(compact_positions, compact_ids)
        if "mm_token_type_ids" in inputs:
            model_inputs["mm_token_type_ids"] = torch.stack(
                [
                    self._replace_groups(
                        inputs["mm_token_type_ids"][row],
                        counts,
                        [inputs["mm_token_type_ids"].new_full((keep,), 1) for keep in keep_counts],
                        config_image_id,
                        reference=inputs["input_ids"][row],
                    )
                    for row in range(inputs["mm_token_type_ids"].shape[0])
                ]
            )
        model_inputs.pop("pixel_values", None)
        model_inputs.pop("pixel_values_videos", None)
        if "pixel_values" in inputs:
            model_inputs["pixel_values"] = inputs["pixel_values"]
        self._last_selection = {
            "retained_tokens": sum(keep_counts),
            "original_tokens": sum(counts),
            "keep_counts": keep_counts,
            "mode": self.pruning_mode,
            "strategy": getattr(self, "pruning_strategy", "topk"),
            "layer": self.pruning_layer,
            "budgets": [selection.budget for selection in selections],
            "indices": [selection.indices.detach().cpu().tolist() for selection in selections],
            "groups": [
                {
                    "final": selection.final_indices.detach().cpu().tolist(),
                    "middle": selection.middle_indices.detach().cpu().tolist(),
                    "shallow": selection.shallow_indices.detach().cpu().tolist(),
                }
                for selection in selections
            ],
        }
        self._last_cached_visual = _SelectedVisionOutput(
            pooler_output=_FeatureTuple(selected_features),
            deepstack_features=[
                _FeatureTuple((branch[image_index] for image_index in range(len(counts))))
                for branch in self._last_native_deepstack
            ],
        )
        return (model_inputs, sum(keep_counts), sum(counts))

    @_inference_mode()
    def answer(
        self, image: Any, prompt: str, *, scoring_text: str | None = None
    ) -> PrunedModelOutput:
        try:
            inputs = self._inputs(image, prompt)
        except Exception as exc:
            raise RuntimeError(
                f"prompt/processor stage failed before visual scoring: {type(exc).__name__}: {exc}"
            ) from exc
        start = time.perf_counter()
        try:
            pruning_started = time.perf_counter()
            model_inputs, image_tokens, original_image_tokens = self._pruned_inputs(
                inputs, scoring_text=scoring_text
            )
            if self.device.type == "cuda":
                torch.cuda.synchronize(self.device)
            pruning_latency = time.perf_counter() - pruning_started
        except Exception as exc:
            raise RuntimeError(
                f"ViT/projector/text-token/pruning stage failed: {type(exc).__name__}: {exc}"
            ) from exc
        input_tokens = int(model_inputs["input_ids"].shape[-1])
        prefill_started = time.perf_counter()
        prefill_holder: dict[str, float] = {}
        hook_handle = None
        register = getattr(self.llm, "register_forward_hook", None)
        if callable(register):

            def _capture_prefill(*_: Any, **__: Any) -> None:
                if "seconds" not in prefill_holder:
                    prefill_holder["seconds"] = time.perf_counter() - prefill_started

            hook_handle = register(_capture_prefill)
        cached_visual = self._last_cached_visual if self.visual_pruning else None
        feature_owners: list[Any] = []
        for owner in (getattr(self.model, "model", None), self.model):
            if (
                owner is not None
                and callable(getattr(owner, "get_image_features", None))
                and all((owner is not existing for existing in feature_owners))
            ):
                feature_owners.append(owner)

        def _cached_features(*_: Any, **__: Any) -> _SelectedVisionOutput:
            if cached_visual is None:
                raise RuntimeError("selected visual features are unavailable for generation")
            return cached_visual

        def _compact_rope_index(*_: Any, **__: Any):
            return (model_inputs["position_ids"], self._last_compact_rope_deltas)

        @contextmanager
        def _cached_feature_scope():
            if cached_visual is None or not feature_owners:
                yield
                return
            with ExitStack() as stack:
                for owner in feature_owners:
                    stack.enter_context(
                        _temporary_method(owner, "get_image_features", _cached_features)
                    )
                    if _is_early_middle_mode(self.pruning_mode) and callable(
                        getattr(owner, "get_rope_index", None)
                    ):
                        stack.enter_context(
                            _temporary_method(owner, "get_rope_index", _compact_rope_index)
                        )
                yield

        use_embeds_fallback = cached_visual is not None and (not feature_owners)
        if use_embeds_fallback:
            selected = torch.cat(cached_visual.pooler_output, dim=0)
            model_inputs["inputs_embeds"] = self._compose_inputs_embeds(
                model_inputs["input_ids"], selected
            )
            model_inputs.pop("pixel_values", None)
            model_inputs.pop("pixel_values_videos", None)
        allowed_inputs = {
            "input_ids",
            "inputs_embeds",
            "attention_mask",
            "position_ids",
            "pixel_values",
            "pixel_values_videos",
            "image_grid_thw",
            "video_grid_thw",
            "mm_token_type_ids",
        }
        model_inputs = {key: value for key, value in model_inputs.items() if key in allowed_inputs}

        def _generate() -> Any:
            return self.model.generate(
                **model_inputs,
                max_new_tokens=self.max_new_tokens,
                min_new_tokens=1,
                do_sample=False,
                use_cache=True,
                return_dict_in_generate=False,
                output_scores=False,
                output_attentions=False,
                output_hidden_states=False,
            )

        generation_started = time.perf_counter()
        try:
            if cached_visual is not None and feature_owners:
                with _cached_feature_scope():
                    generated = _generate()
            else:
                generated = _generate()
        except Exception as exc:
            input_summary = {
                key: (
                    f"{type(value).__name__}{tuple(value.shape)}"
                    if torch.is_tensor(value)
                    else type(value).__name__
                )
                for key, value in model_inputs.items()
            }
            raise RuntimeError(
                f"LLM generation stage failed: {type(exc).__name__}: {exc}; model_inputs={input_summary}"
            ) from exc
        finally:
            if hook_handle is not None:
                hook_handle.remove()
            self._last_cached_visual = None
            self._last_native_final_features = []
            self._last_native_deepstack = []
            self._last_early_selections = []
        if self.device.type == "cuda":
            torch.cuda.synchronize(self.device)
        generation_seconds = time.perf_counter() - generation_started
        if hasattr(generated, "sequences"):
            generated = generated.sequences
        if generated.ndim == 1:
            generated = generated.unsqueeze(0)
        prompt_ids = model_inputs["input_ids"]
        prompt_length = int(prompt_ids.shape[-1])
        has_prompt_prefix = (
            generated.ndim == 2
            and int(generated.shape[-1]) >= prompt_length
            and torch.equal(
                generated[:, :prompt_length],
                prompt_ids.to(device=generated.device, dtype=generated.dtype),
            )
        )
        new_ids = generated[:, prompt_length:] if has_prompt_prefix else generated
        decode_started = time.perf_counter()
        text = self.processor.batch_decode(
            new_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False
        )[0]
        text = _normalize_model_answer(text)
        output_tokens = (
            int(new_ids.shape[-1]) if torch.is_tensor(new_ids) else int(new_ids[0].numel())
        )
        if not text:
            raise RuntimeError(
                f"the model returned no decodable answer tokens; generated_shape={tuple(generated.shape)}, continuation_tokens={output_tokens}"
            )
        decode_seconds = time.perf_counter() - decode_started
        seconds = time.perf_counter() - start
        prefill_latency = float(prefill_holder.get("seconds", seconds))
        return PrunedModelOutput(
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
                vision_prune_layer=(
                    self.pruning_layer
                    if self.visual_pruning and _is_early_middle_mode(self.pruning_mode)
                    else None
                ),
            ),
            original_image_tokens=original_image_tokens,
            pruned_image_tokens=image_tokens,
            retained_tokens=image_tokens,
            retention_ratio=(
                float(image_tokens) / float(original_image_tokens) if original_image_tokens else 0.0
            ),
            pruning_latency=pruning_latency,
            prefill_latency=prefill_latency,
            pruning_seconds=pruning_latency,
            generation_seconds=generation_seconds,
            decode_seconds=decode_seconds,
            selection_budgets=list(self._last_selection.get("budgets", [])),
            selected_indices=list(self._last_selection.get("indices", [])),
            selection_groups=list(self._last_selection.get("groups", [])),
            selection_mode=str(self._last_selection.get("mode", self.pruning_mode)),
            selection_strategy=str(
                self._last_selection.get("strategy", getattr(self, "pruning_strategy", "topk"))
            ),
            selection_layer=self._last_selection.get("layer"),
            vision_layer_token_counts=(
                list(self._last_vision_layer_token_counts)
                if self.visual_pruning and _is_early_middle_mode(self.pruning_mode)
                else []
            ),
            vision_merge_unit=(
                int(_get(_get(self.config, "vision_config", None), "spatial_merge_size", 1) or 1)
                ** 2
                if _is_early_middle_mode(self.pruning_mode)
                else 0
            ),
        )

    @staticmethod
    def _raw_indices(
        selections: Sequence[Any], raw_counts: Sequence[int], merge_unit: int, device: Any
    ) -> tuple[list[torch.Tensor], torch.Tensor]:
        parts: list[torch.Tensor] = []
        global_parts: list[torch.Tensor] = []
        offset = 0
        for selection, raw_count in zip(selections, raw_counts):
            groups = selection.indices.to(device=device, dtype=torch.long)
            offsets = torch.arange(merge_unit, device=device, dtype=torch.long)
            local = (groups.unsqueeze(-1) * merge_unit + offsets).reshape(-1)
            parts.append(local)
            global_parts.append(local + offset)
            offset += int(raw_count)
        return (parts, torch.cat(global_parts, dim=0))


__all__ = ["PrunedQwenVL", "PrunedModelOutput", "valid_text_token_mask"]
