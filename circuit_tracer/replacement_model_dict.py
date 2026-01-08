import warnings
from collections import defaultdict
from contextlib import contextmanager
from functools import partial
from collections.abc import Callable, Sequence

import torch
import torch.nn.functional as F
from torch import nn
from transformer_lens import HookedTransformer, HookedTransformerConfig
from transformer_lens.hook_points import HookPoint
from transformers.tokenization_utils_base import PreTrainedTokenizerBase

from circuit_tracer.attribution.context import AttributionContext
from circuit_tracer.transcoder import TranscoderSet
from circuit_tracer.transcoder.cross_layer_transcoder import CrossLayerTranscoder
from circuit_tracer.utils import get_default_device
from circuit_tracer.utils.hf_utils import load_transcoder_from_hub

# Type definition for an intervention tuple (layer, position, feature_idx, value)
Intervention = tuple[
    int | torch.Tensor, int | slice | torch.Tensor, int | torch.Tensor, float | torch.Tensor
]


class ReplacementMLP(nn.Module):
    """Wrapper for a TransformerLens MLP layer that adds in extra hooks"""

    def __init__(self, old_mlp: nn.Module):
        super().__init__()
        self.old_mlp = old_mlp
        self.hook_in = HookPoint()
        self.hook_out = HookPoint()

    def forward(self, x):
        x = self.hook_in(x)
        mlp_out = self.old_mlp(x)
        return self.hook_out(mlp_out)


class ReplacementUnembed(nn.Module):
    """Wrapper for a TransformerLens Unembed layer that adds in extra hooks"""

    def __init__(self, old_unembed: nn.Module):
        super().__init__()
        self.old_unembed = old_unembed
        self.hook_pre = HookPoint()
        self.hook_post = HookPoint()

    @property
    def W_U(self):
        return self.old_unembed.W_U

    @property
    def b_U(self):
        return self.old_unembed.b_U

    def forward(self, x):
        x = self.hook_pre(x)
        x = self.old_unembed(x)
        return self.hook_post(x)


class ReplacementModel(HookedTransformer):
    feature_input_hook: str
    feature_output_hook: str
    skip_transcoder: bool
    scan: str | list[str] | None
    tokenizer: PreTrainedTokenizerBase

    @classmethod
    def from_config(
        cls,
        config: HookedTransformerConfig,
        **kwargs,
    ) -> "ReplacementModel":
        """Create a ReplacementModel from a given HookedTransformerConfig

        Args:
            config (HookedTransformerConfig): the config of the HookedTransformer

        Returns:
            ReplacementModel: The loaded ReplacementModel
        """
        model = cls(config, **kwargs)
        model._configure_replacement_model()
        return model

    @classmethod
    def from_pretrained(
        cls,
        model_name: str,
        transcoders: TranscoderSet | CrossLayerTranscoder,  # Accept both
        **kwargs,
    ) -> "ReplacementModel":
        """Create a ReplacementModel from the name of HookedTransformer and TranscoderSet

        Args:
            model_name (str): the name of the pretrained HookedTransformer
            transcoders (TranscoderSet): The transcoder set with configuration

        Returns:
            ReplacementModel: The loaded ReplacementModel
        """
        model = super().from_pretrained(
            model_name,
            fold_ln=False,
            center_writing_weights=False,
            center_unembed=False,
            **kwargs,
        )

        model._configure_replacement_model()
        return model


    def _configure_replacement_model(self):
        self.feature_input_hook = transcoder_set.feature_input_hook
        self.original_feature_output_hook = transcoder_set.feature_output_hook
        self.feature_output_hook = transcoder_set.feature_output_hook + ".hook_out_grad"


        for block in self.blocks:
            block.mlp = ReplacementMLP(block.mlp)  # type: ignore

        self.unembed = ReplacementUnembed(self.unembed)

        self._configure_gradient_flow()
        self._deduplicate_attention_buffers()
        self.setup()

    def _configure_gradient_flow(self):
        if isinstance(self.transcoders, TranscoderSet):
            for layer, transcoder in enumerate(self.transcoders):
                self._configure_skip_connection(self.blocks[layer], transcoder)
        else:
            for layer in range(self.cfg.n_layers):
                self._configure_skip_connection(self.blocks[layer], self.transcoders)

        def stop_gradient(acts, hook):
            return acts.detach()

        for block in self.blocks:
            block.attn.hook_pattern.add_hook(stop_gradient, is_permanent=True)  # type: ignore
            block.ln1.hook_scale.add_hook(stop_gradient, is_permanent=True)  # type: ignore
            block.ln2.hook_scale.add_hook(stop_gradient, is_permanent=True)  # type: ignore
            if hasattr(block, "ln1_post"):
                block.ln1_post.hook_scale.add_hook(stop_gradient, is_permanent=True)  # type: ignore
            if hasattr(block, "ln2_post"):
                block.ln2_post.hook_scale.add_hook(stop_gradient, is_permanent=True)  # type: ignore
            self.ln_final.hook_scale.add_hook(stop_gradient, is_permanent=True)  # type: ignore

        for param in self.parameters():
            param.requires_grad = False

        def enable_gradient(tensor, hook):
            tensor.requires_grad = True
            return tensor

        self.hook_embed.add_hook(enable_gradient, is_permanent=True)

    def _configure_skip_connection(self, block, transcoder):
        cached = {}

        def cache_activations(acts, hook):
            cached["acts"] = acts

        def add_skip_connection(acts: torch.Tensor, hook: HookPoint, grad_hook: HookPoint):
            # We add grad_hook because we need a way to hook into the gradients of the output
            # of this function. If we put the backwards hook here at hook, the grads will be 0
            # because we detached acts.
            skip_input_activation = cached.pop("acts")
            if hasattr(transcoder, "W_skip") and transcoder.W_skip is not None:
                skip = transcoder.compute_skip(skip_input_activation)
            else:
                skip = skip_input_activation * 0
            return grad_hook(skip + (acts - skip).detach())

        # add feature input hook
        output_hook_parts = self.feature_input_hook.split(".")
        subblock = block
        for part in output_hook_parts:
            subblock = getattr(subblock, part)
        subblock.add_hook(cache_activations, is_permanent=True)

        # add feature output hook and special grad hook
        output_hook_parts = self.original_feature_output_hook.split(".")
        subblock = block
        for part in output_hook_parts:
            subblock = getattr(subblock, part)
        subblock.hook_out_grad = HookPoint()
        subblock.add_hook(
            partial(add_skip_connection, grad_hook=subblock.hook_out_grad),
            is_permanent=True,
        )

    def _deduplicate_attention_buffers(self):
        """
        Share attention buffers across layers to save memory.

        TransformerLens makes separate copies of the same masks and RoPE
        embeddings for each layer - This just keeps one copy
        of each and shares it across all layers.
        """

        attn_masks = {}

        for block in self.blocks:
            attn_masks[block.attn.attn_type] = block.attn.mask  # type: ignore
            if hasattr(block.attn, "rotary_sin"):
                attn_masks["rotary_sin"] = block.attn.rotary_sin  # type: ignore
                attn_masks["rotary_cos"] = block.attn.rotary_cos  # type: ignore

        for block in self.blocks:
            block.attn.mask = attn_masks[block.attn.attn_type]  # type: ignore
            if hasattr(block.attn, "rotary_sin"):
                block.attn.rotary_sin = attn_masks["rotary_sin"]  # type: ignore
                block.attn.rotary_cos = attn_masks["rotary_cos"]  # type: ignore

    def _get_activation_caching_hooks(
        self,
        sparse: bool = False,
        apply_activation_function: bool = True,
        append: bool = False,
    ) -> tuple[list[torch.Tensor], list[tuple[str, Callable]]]:
        activation_matrix = (
            [[] for _ in range(self.cfg.n_layers)] if append else [None] * self.cfg.n_layers
        )

        def cache_activations(acts, hook, layer):
            transcoder_acts = (
                self.transcoders.encode_layer(
                    acts, layer, apply_activation_function=apply_activation_function
                )
                .detach()
                .squeeze(0)
            )
            if sparse:
                transcoder_acts = transcoder_acts.to_sparse()

            if append:
                activation_matrix[layer].append(transcoder_acts)
            else:
                activation_matrix[layer] = transcoder_acts  # type: ignore

        activation_hooks = [
            (
                f"blocks.{layer}.{self.feature_input_hook}",
                partial(cache_activations, layer=layer),
            )
            for layer in range(self.cfg.n_layers)
        ]
        return activation_matrix, activation_hooks  # type: ignore

    def get_activations(
        self,
        inputs: str | torch.Tensor,
        sparse: bool = False,
        apply_activation_function: bool = True,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Get the transcoder activations for a given prompt

        Args:
            inputs (str | torch.Tensor): The inputs you want to get activations over
            sparse (bool, optional): Whether to return a sparse tensor of activations.
                Useful if d_transcoder is large. Defaults to False.

        Returns:
            tuple[torch.Tensor, torch.Tensor]: the model logits on the inputs and the
                associated activation cache
        """

        activation_cache, activation_hooks = self._get_activation_caching_hooks(
            sparse=sparse,
            apply_activation_function=apply_activation_function,
        )
        with torch.inference_mode(), self.hooks(activation_hooks):  # type: ignore
            logits = self(inputs)
        activation_cache = torch.stack(activation_cache)
        if sparse:
            activation_cache = activation_cache.coalesce()
        return logits, activation_cache

    @contextmanager
    def zero_softcap(self):
        current_softcap = self.cfg.output_logits_soft_cap
        try:
            self.cfg.output_logits_soft_cap = 0.0
            yield
        finally:
            self.cfg.output_logits_soft_cap = current_softcap

    def ensure_tokenized(self, prompt: str | torch.Tensor | list[int]) -> torch.Tensor:
        """Convert prompt to 1-D tensor of token ids with proper special token handling.

        This method ensures that a special token (BOS/PAD) is prepended to the input sequence.
        The first token position in transformer models typically exhibits unusually high norm
        and an excessive number of active features due to how models process the beginning of
        sequences. By prepending a special token, we ensure that actual content tokens have
        more consistent and interpretable feature activations, avoiding the artifacts present
        at position 0. This prepended token is later ignored during attribution analysis.

        Args:
            prompt: String, tensor, or list of token ids representing a single sequence

        Returns:
            1-D tensor of token ids with BOS/PAD token at the beginning

        Raises:
            TypeError: If prompt is not str, tensor, or list
            ValueError: If tensor has wrong shape (must be 1-D or 2-D with batch size 1)
        """

        if isinstance(prompt, str):
            tokens = self.tokenizer(prompt, return_tensors="pt").input_ids.squeeze(0)
        elif isinstance(prompt, torch.Tensor):
            tokens = prompt.squeeze()
        elif isinstance(prompt, list):
            tokens = torch.tensor(prompt, dtype=torch.long).squeeze()
        else:
            raise TypeError(f"Unsupported prompt type: {type(prompt)}")

        if tokens.ndim > 1:
            raise ValueError(f"Tensor must be 1-D, got shape {tokens.shape}")

        # Check if a special token is already present at the beginning
        if tokens[0] in self.tokenizer.all_special_ids:
            return tokens.to(self.cfg.device)

        # Prepend a special token to avoid artifacts at position 0
        candidate_bos_token_ids = [
            self.tokenizer.bos_token_id,
            self.tokenizer.pad_token_id,
            self.tokenizer.eos_token_id,
        ]
        candidate_bos_token_ids += self.tokenizer.all_special_ids

        dummy_bos_token_id = next(filter(None, candidate_bos_token_ids))
        if dummy_bos_token_id is None:
            warnings.warn(
                "No suitable special token found for BOS token replacement. "
                "The first token will be ignored."
            )
        else:
            tokens = torch.cat([torch.tensor([dummy_bos_token_id], device=tokens.device), tokens])

        return tokens.to(self.cfg.device)

    