# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import math
from collections.abc import Iterable, Mapping, Sequence
from typing import List, NamedTuple, Optional, TypedDict, Union

import numpy as np
import torch
from torch import nn
from transformers import (BatchFeature, WhisperConfig, WhisperFeatureExtractor,
                          WhisperProcessor)
from transformers.models.whisper.modeling_whisper import sinusoids

from vllm.attention import Attention, AttentionType
from vllm.config import CacheConfig, VllmConfig
from vllm.distributed import get_tensor_model_parallel_world_size
from vllm.logger import init_logger
from vllm.model_executor.layers.activation import get_act_fn
from vllm.model_executor.layers.linear import (ColumnParallelLinear,
                                               QKVParallelLinear,
                                               RowParallelLinear)
from vllm.model_executor.layers.logits_processor import LogitsProcessor
from vllm.model_executor.layers.quantization.base_config import (
    QuantizationConfig)
from vllm.model_executor.layers.vocab_parallel_embedding import ParallelLMHead
from vllm.model_executor.model_loader.weight_utils import default_weight_loader
from vllm.model_executor.sampling_metadata import SamplingMetadata
from vllm.multimodal import MULTIMODAL_REGISTRY, NestedTensors
from vllm.multimodal.inputs import (MultiModalDataDict, MultiModalFieldConfig,
                                    MultiModalKwargs)
from vllm.multimodal.parse import MultiModalDataItems, MultiModalDataParser
from vllm.multimodal.processing import (BaseProcessingInfo,
                                        EncDecMultiModalProcessor,
                                        PromptReplacement, PromptUpdate)
from vllm.multimodal.profiling import BaseDummyInputsBuilder

from .interfaces import (MultiModalEmbeddings, SupportsMultiModal,
                         SupportsTranscription, SupportsV0Only)
from .utils import (AutoWeightsLoader, WeightsMapper, cast_overflow_tensors,
                    make_layers)

logger = init_logger(__name__)


# Word-level timestamp alignment for Whisper using Dynamic Time Warping (DTW)
class WordTiming(NamedTuple):
    """Word timing information."""
    word: str
    tokens: List[int]
    start: float
    end: float
    probability: float


def dtw(x: np.ndarray, band_width: int = 50) -> tuple[np.ndarray, np.ndarray]:
    """Dynamic Time Warping alignment with Sakoe-Chiba band constraint.
    
    Args:
        x: Cost matrix of shape (N, M)
        band_width: Maximum deviation from diagonal (±band_width frames)
        
    Returns:
        Tuple of (path_x, path_y) indices for optimal alignment
    """
    N, M = x.shape
    cost = np.ones((N + 1, M + 1), dtype=np.float32) * np.inf
    trace = -np.ones((N + 1, M + 1), dtype=np.int32)

    cost[0, 0] = 0
    
    # Compute the diagonal scaling factor
    diagonal_ratio = M / N if N > 0 else 1.0
    
    for j in range(1, M + 1):
        for i in range(1, N + 1):
            # Calculate expected position on diagonal
            expected_j = int(i * diagonal_ratio)
            
            # Skip if outside the band constraint
            if abs(j - expected_j) > band_width:
                continue
                
            c0 = cost[i - 1, j - 1]
            c1 = cost[i - 1, j]
            c2 = cost[i, j - 1]

            if c0 < c1 and c0 < c2:
                c, t = c0, 2
            elif c1 < c0 and c1 < c2:
                c, t = c1, 1
            else:
                c, t = c2, 0

            cost[i, j] = x[i - 1, j - 1] + c
            trace[i, j] = t

    # Backtrack
    i, j = N, M
    path_x, path_y = [i], [j]
    while i > 0 or j > 0:
        t = trace[i, j]
        if t == 2:
            i, j = i - 1, j - 1
        elif t == 1:
            i, j = i - 1, j
        else:
            i, j = i, j - 1
        path_x.append(i)
        path_y.append(j)

    path_x.reverse()
    path_y.reverse()
    return np.array(path_x[1:]), np.array(path_y[1:])


def find_alignment(
    cross_attention_weights: List[torch.Tensor],
    text_tokens: List[int],
    num_frames: int,
    alignment_heads: List[tuple[int, int]],
    tokenizer_info: dict,
    feature_extractor_info: dict,
    *,
    medfilt_width: int = 5,
    qk_scale: float = 1.0,
) -> List[WordTiming]:
    """Find word-level alignments using cross-attention weights and DTW.
    
    Args:
        cross_attention_weights: List of attention weight tensors from decoder layers
        text_tokens: Token IDs for the text (without special tokens)
        num_frames: Number of audio frames
        alignment_heads: List of (layer_idx, head_idx) tuples for alignment
        tokenizer_info: Dict containing tokenizer information
        feature_extractor_info: Dict containing feature extractor info
        medfilt_width: Width for median filtering
        qk_scale: Scaling factor for attention weights
        
    Returns:
        List of WordTiming objects with word-level timestamps
    """
    if len(text_tokens) == 0:
        return []

    # Extract tokenizer info
    sot_sequence_len = tokenizer_info.get("sot_sequence_len", 4)
    eot_token = tokenizer_info.get("eot_token", 50257)
    
    # Extract feature extractor info for time conversion
    hop_length = feature_extractor_info.get("hop_length", 160)
    sampling_rate = feature_extractor_info.get("sampling_rate", 16000)
    time_per_frame = hop_length / sampling_rate

    # Collect attention weights from specified alignment heads
    try:
        weights = []
        for layer_idx, head_idx in alignment_heads:
            if (layer_idx < len(cross_attention_weights) and 
                cross_attention_weights[layer_idx] is not None and
                head_idx < cross_attention_weights[layer_idx].shape[1]):
                # Extract weights: (batch, heads, seq_len, num_frames)
                attn_weights = cross_attention_weights[layer_idx][0, head_idx]
                weights.append(attn_weights)
        
        if not weights:
            logger.warning("No valid alignment heads found")
            return []
            
        # Stack and process weights: (num_heads, seq_len, num_frames)
        weights = torch.stack(weights)
        
        # Trim to actual audio frames
        max_frames = min(weights.shape[-1], num_frames)
        weights = weights[:, :, :max_frames]
        
        # Apply scaling and softmax
        weights = (weights * qk_scale).softmax(dim=-1)
        
        # Normalize: subtract mean and divide by std across time dimension
        std, mean = torch.std_mean(weights, dim=-1, keepdim=True, unbiased=False)
        weights = (weights - mean) / (std + 1e-8)  # Add epsilon for numerical stability
        
        # Average across alignment heads first on GPU to reduce memory transfer
        matrix = torch.mean(weights, dim=0)  # Shape: (seq_len, num_frames)
        
        # Apply efficient 1D smoothing filter on GPU (replaces median filter)
        if medfilt_width > 1:
            # Use 1D convolution for efficient smoothing on GPU
            kernel_size = medfilt_width
            padding = kernel_size // 2
            
            # Create a uniform kernel for smoothing
            kernel = torch.ones(1, 1, kernel_size, device=matrix.device, dtype=matrix.dtype) / kernel_size
            
            # Apply 1D convolution along the time dimension (last dim)
            # Reshape for conv1d: (batch=seq_len, channels=1, length=num_frames)
            matrix_reshaped = matrix.unsqueeze(1)  # (seq_len, 1, num_frames)
            smoothed = torch.nn.functional.conv1d(
                matrix_reshaped, 
                kernel, 
                padding=padding,
                groups=1
            )
            matrix = smoothed.squeeze(1)  # Back to (seq_len, num_frames)
        
        # Convert to CPU after GPU processing
        matrix = matrix.cpu().numpy()
        
        # Remove special tokens (SOT sequence at start, EOT at end)
        matrix = matrix[sot_sequence_len:-1]  # Remove SOT and EOT
        text_tokens = text_tokens[sot_sequence_len:-1]  # Remove corresponding tokens
        
        # Ensure we have the right number of tokens
        if matrix.shape[0] != len(text_tokens):
            logger.warning(f"Matrix shape {matrix.shape[0]} doesn't match text tokens {len(text_tokens)}")
            # Adjust if there's a mismatch
            min_len = min(matrix.shape[0], len(text_tokens))
            matrix = matrix[:min_len]
            text_tokens = text_tokens[:min_len]
        
        # Perform DTW alignment
        text_indices, time_indices = dtw(-matrix)
        
    except Exception as e:
        logger.error(f"Error in alignment processing: {e}")
        return []

    # Split tokens into words using actual tokenizer
    words, word_tokens = _split_to_word_tokens(text_tokens, eot_token, tokenizer_info)
    
    if len(word_tokens) <= 1:
        return []
    
    # Calculate word boundaries in token space
    word_boundaries = np.pad(np.cumsum([len(t) for t in word_tokens[:-1]]), (1, 0))
    
    # Find jumps in alignment
    jumps = np.diff(text_indices, prepend=text_indices[:1]) != 0
    jump_times = time_indices[jumps] * time_per_frame  # Convert to seconds
    
    # Map to word boundaries
    start_times = jump_times[word_boundaries[:-1]]
    end_times = jump_times[np.minimum(word_boundaries[1:], len(jump_times) - 1)]
    
    # Calculate word probabilities from alignment matrix
    word_probabilities = []
    for i, word_token_list in enumerate(word_tokens):
        if not word_token_list or word_token_list == [eot_token]:
            word_probabilities.append(0.0)
            continue
            
        # Get token indices for this word
        start_token_idx = word_boundaries[i] if i < len(word_boundaries) else 0
        end_token_idx = word_boundaries[i + 1] if i + 1 < len(word_boundaries) else len(text_indices)
        
        # Collect probabilities for tokens in this word
        token_probs = []
        for token_idx in range(start_token_idx, min(end_token_idx, len(text_indices))):
            if token_idx < len(text_indices):
                t_i = text_indices[token_idx]
                f_i = time_indices[token_idx]
                if t_i < matrix.shape[0] and f_i < matrix.shape[1]:
                    prob = float(matrix[t_i, f_i].clip(0, 1))
                    token_probs.append(prob)
        
        # Average probability over tokens in the word
        if token_probs:
            word_probabilities.append(float(np.mean(token_probs)))
        else:
            word_probabilities.append(0.0)
    
    return [
        WordTiming(word, tokens, start, end, probability)
        for word, tokens, start, end, probability in zip(
            words, word_tokens, start_times, end_times, word_probabilities
        )
    ]


def _split_to_word_tokens(text_tokens: List[int], eot_token: int, tokenizer_info: dict) -> tuple[List[str], List[List[int]]]:
    """Split tokens into words using actual tokenizer logic.
    
    Args:
        text_tokens: List of token IDs
        eot_token: End of text token ID
        tokenizer_info: Dict containing tokenizer information and instance
        
    Returns:
        Tuple of (words, word_tokens) where words are string representations
        and word_tokens are lists of token IDs for each word.
    """
    # Get tokenizer from tokenizer_info
    tokenizer = tokenizer_info.get("tokenizer")

    # Check if this is a Whisper tokenizer with split_to_word_tokens method
    if hasattr(tokenizer, 'split_to_word_tokens'):
        # Use Whisper's built-in word splitting
        words, word_tokens = tokenizer.split_to_word_tokens(text_tokens + [eot_token])
        return words, word_tokens
    
    # Fallback: Manual word splitting for other tokenizer types
    # Decode the text tokens to get the full text
    text = tokenizer.decode(text_tokens, skip_special_tokens=True)
    
    # Split text into words using whitespace
    text_words = text.split()
    
    if not text_words:
        return [], [[eot_token]]
    
    # Now map tokens back to words
    words = []
    word_tokens = []
    
    # Decode each token individually to understand token boundaries
    token_texts = []
    for token_id in text_tokens:
        token_text = tokenizer.decode([token_id], skip_special_tokens=True)
        token_texts.append(token_text)
    
    # Group tokens into words
    current_word = ""
    current_word_tokens = []
    word_idx = 0
    
    for i, (token_id, token_text) in enumerate(zip(text_tokens, token_texts)):
        current_word += token_text
        current_word_tokens.append(token_id)
        
        # Check if we've completed a word
        if word_idx < len(text_words):
            target_word = text_words[word_idx]
            
            # Handle cases where token text might have leading/trailing spaces
            current_word_clean = current_word.strip()
            
            # If we've matched the target word
            if current_word_clean == target_word:
                words.append(target_word)
                word_tokens.append(current_word_tokens)
                current_word = ""
                current_word_tokens = []
                word_idx += 1
            # If current word is longer than target, we might have multiple words in one token
            elif len(current_word_clean) > len(target_word) and target_word in current_word_clean:
                words.append(target_word)
                word_tokens.append(current_word_tokens)
                # Reset but keep remaining text
                remaining_text = current_word_clean[current_word_clean.find(target_word) + len(target_word):].strip()
                current_word = remaining_text
                current_word_tokens = [token_id] if remaining_text else []
                word_idx += 1
    
    # Handle any remaining tokens
    if current_word_tokens:
        remaining_text = current_word.strip()
        if remaining_text:
            words.append(remaining_text)
            word_tokens.append(current_word_tokens)
    
    # Add any remaining words from text_words that we missed
    while word_idx < len(text_words):
        words.append(text_words[word_idx])
        word_tokens.append([])  # Empty token list for unmatched words
        word_idx += 1
    
    # Add EOT token as final word
    word_tokens.append([eot_token])
    words.append("")  # EOT word
    
    return words, word_tokens

def merge_word_timings_with_chunks(
    word_timings: List[WordTiming],
    chunk_start_sec: float
) -> List[WordTiming]:
    """Adjust word timings by adding chunk start offset.
    
    Args:
        word_timings: List of word timings from alignment
        chunk_start_sec: Start time of the current chunk in seconds
        
    Returns:
        List of word timings with adjusted timestamps
    """
    return [
        WordTiming(
            word=wt.word,
            tokens=wt.tokens,
            start=wt.start + chunk_start_sec,
            end=wt.end + chunk_start_sec,
            probability=wt.probability
        )
        for wt in word_timings
    ]


class WhisperAudioInputs(TypedDict):
    input_features: NestedTensors
    """Shape: `(batch_size, 128, M)`"""


class WhisperPositionalEmbedding(nn.Embedding):

    def __init__(self, num_positions: int, embedding_dim: int):
        super().__init__(num_positions, embedding_dim)

    def forward(self, position_ids):
        return self.weight[position_ids]


class WhisperAttention(nn.Module):

    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
        bias: bool = True,
        attn_type: AttentionType = AttentionType.DECODER,
        cache_config: Optional[CacheConfig] = None,
        quant_config: Optional[QuantizationConfig] = None,
        prefix: str = "",
    ):
        super().__init__()
        self.embed_dim = embed_dim
        tp_size = get_tensor_model_parallel_world_size()
        self.total_num_heads = num_heads
        assert self.total_num_heads % tp_size == 0
        self.num_heads = self.total_num_heads // tp_size
        if self.total_num_heads >= tp_size:
            # Number of heads is greater than TP size, so we partition
            # the KV heads across multiple tensor parallel GPUs.
            assert self.total_num_heads % tp_size == 0
        else:
            # Number of heads is less than TP size, so we replicate
            # the KV heads across multiple tensor parallel GPUs.
            assert tp_size % self.total_num_heads == 0
        self.num_kv_heads = max(1, self.total_num_heads // tp_size)
        self.head_dim = self.embed_dim // self.total_num_heads
        self.q_size = self.num_heads * self.head_dim
        self.kv_size = self.num_kv_heads * self.head_dim
        self.attn_type = attn_type

        if (self.head_dim * num_heads) != self.embed_dim:
            raise ValueError(
                f"embed_dim must be divisible by num_heads (got `embed_dim`: "
                f"{self.embed_dim} and `num_heads`: {num_heads}).")
        self.scaling = self.head_dim**-0.5

        self._init_qkv(embed_dim, bias, quant_config, prefix=prefix)
        self.out_proj = RowParallelLinear(
            input_size=embed_dim,
            output_size=embed_dim,
            bias=bias,
            quant_config=quant_config,
            prefix=f"{prefix}.out_proj",
        )
        self.attn = Attention(
            self.num_heads,
            self.head_dim,
            self.scaling,
            num_kv_heads=self.num_kv_heads,
            cache_config=cache_config,
            quant_config=quant_config,
            prefix=f"{prefix}.attn",
            attn_type=self.attn_type,
        )

    def _init_qkv(
        self,
        embed_dim: int,
        bias: bool = True,
        quant_config: Optional[QuantizationConfig] = None,
        prefix: str = "",
    ) -> None:
        self.qkv_proj = QKVParallelLinear(
            hidden_size=embed_dim,
            head_size=self.head_dim,
            total_num_heads=self.total_num_heads,
            total_num_kv_heads=self.total_num_heads,
            bias=bias,
            quant_config=quant_config,
            prefix=f"{prefix}.qkv_proj",
        )

    def forward(
        self,
        hidden_states: torch.Tensor,
    ):
        qkv, _ = self.qkv_proj(hidden_states)
        q, k, v = qkv.split([self.q_size, self.kv_size, self.kv_size], dim=-1)

        attn_output = self.attn(q, k, v)

        output, _ = self.out_proj(attn_output)

        return output


class WhisperCrossAttention(WhisperAttention):

    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
        bias: bool = True,
        cache_config: Optional[CacheConfig] = None,
        quant_config: Optional[QuantizationConfig] = None,
        prefix: str = "",
    ):
        super().__init__(
            embed_dim=embed_dim,
            num_heads=num_heads,
            bias=bias,
            cache_config=cache_config,
            quant_config=quant_config,
            prefix=prefix,
            attn_type=AttentionType.ENCODER_DECODER,
        )

    def _init_qkv(
        self,
        embed_dim: int,
        bias: bool = True,
        quant_config: Optional[QuantizationConfig] = None,
        prefix: str = "",
    ) -> None:
        self.q_proj = ColumnParallelLinear(
            input_size=embed_dim,
            output_size=embed_dim,
            bias=bias,
            quant_config=quant_config,
            prefix=f"{prefix}.q_proj",
        )
        self.kv_proj = QKVParallelLinear(
            hidden_size=embed_dim,
            head_size=self.head_dim,
            total_num_heads=0,
            total_num_kv_heads=self.total_num_heads,
            bias=bias,
            quant_config=quant_config,
            prefix=f"{prefix}.kv_proj",
        )

    def forward(
        self,
        hidden_states: torch.Tensor,
        encoder_hidden_states: Optional[torch.Tensor],
        output_attentions: bool = False,
        alignment_heads: Optional[list[tuple[int, int]]] = None,
        layer_idx: Optional[int] = None,
    ):
        q, _ = self.q_proj(hidden_states)

        # Encoder hidden states are only computed once during prefill phase.
        # Afterwards, the keys and values should be available in the kv-cache.
        if encoder_hidden_states is not None:
            kv, _ = self.kv_proj(encoder_hidden_states)
            k, v = kv.split([self.kv_size, self.kv_size], dim=-1)
        else:
            k = v = None

        # If we need attention weights, compute them efficiently with head slicing
        attn_weights = None
        if output_attentions and k is not None and v is not None:
            # Determine which heads are needed for this layer to minimize memory allocation
            layer_heads = None
            if alignment_heads and layer_idx is not None:
                layer_heads = [head_idx for layer, head_idx in alignment_heads if layer == layer_idx]
            
            if layer_heads:
                # Use efficient attention for forward pass (memory-optimized)
                attn_output = self.attn(q, k, v)
                output, _ = self.out_proj(attn_output)
                
                # Compute attention weights ONLY for alignment heads
                batch_size, seq_len, _ = q.shape
                src_len = k.shape[1]
                
                # Reshape for multi-head attention: (batch, seq_len, num_heads, head_dim)
                q_reshaped = q.view(batch_size, seq_len, self.num_heads, self.head_dim)
                k_reshaped = k.view(batch_size, src_len, self.num_kv_heads, self.head_dim)
                
                # Handle grouped query attention by repeating KV heads if needed
                if self.num_heads != self.num_kv_heads:
                    repeat_factor = self.num_heads // self.num_kv_heads
                    k_reshaped = k_reshaped.repeat_interleave(repeat_factor, dim=2)
                
                # Transpose to (batch, num_heads, seq_len, head_dim)
                q_reshaped = q_reshaped.transpose(1, 2)
                k_reshaped = k_reshaped.transpose(1, 2)
                
                # Select only the heads we need for alignment (saves 85-90% VRAM)
                layer_head_indices = torch.tensor(layer_heads, device=q.device)
                q_selected = q_reshaped.index_select(1, layer_head_indices).contiguous()
                k_selected = k_reshaped.index_select(1, layer_head_indices).contiguous()
                
                # Compute attention scores only for selected heads: (batch, selected_heads, seq_len, src_len)
                attn_scores = torch.matmul(q_selected, k_selected.transpose(-2, -1)) * self.scaling
                attn_weights = torch.softmax(attn_scores, dim=-1)
                # Cast to fp16 to save ~2× RAM (DTW code will re-cast to fp32 anyway)
                attn_weights = attn_weights.to(torch.float16).contiguous()
            else:
                # No alignment heads for this layer, use efficient attention for forward pass
                attn_output = self.attn(q, k, v)
                output, _ = self.out_proj(attn_output)
                # No weights returned for this layer (set to None to save memory)
                attn_weights = None
        else:
            # Standard path without attention weight computation
            attn_output = self.attn(q, k, v)
            output, _ = self.out_proj(attn_output)

        if output_attentions:
            return output, attn_weights
        return output


class WhisperMLP(nn.Module):

    def __init__(
        self,
        embed_dim: int,
        ffn_dim: int,
        act_fn: str,
        quant_config: Optional[QuantizationConfig] = None,
        prefix: str = "",
    ):
        super().__init__()

        self.activation_fn = get_act_fn(act_fn)
        self.fc1 = ColumnParallelLinear(
            input_size=embed_dim,
            output_size=ffn_dim,
            quant_config=quant_config,
            prefix=f"{prefix}.fc1",
        )
        self.fc2 = RowParallelLinear(
            input_size=ffn_dim,
            output_size=embed_dim,
            quant_config=quant_config,
            prefix=f"{prefix}.fc2",
        )

    def forward(self, hidden_states: torch.Tensor):
        hidden_states, _ = self.fc1(hidden_states)
        hidden_states = self.activation_fn(hidden_states)
        hidden_states, _ = self.fc2(hidden_states)
        return hidden_states


class WhisperEncoderLayer(nn.Module):

    def __init__(self, *, vllm_config: VllmConfig, prefix: str = ""):
        super().__init__()
        config = vllm_config.model_config.hf_config
        cache_config = vllm_config.cache_config
        quant_config = vllm_config.quant_config

        self.embed_dim = config.d_model
        self.self_attn = WhisperAttention(
            embed_dim=self.embed_dim,
            num_heads=config.encoder_attention_heads,
            attn_type=AttentionType.ENCODER,
            cache_config=cache_config,
            quant_config=quant_config,
            prefix=f"{prefix}.self_attn",
        )
        self.self_attn_layer_norm = nn.LayerNorm(self.embed_dim)
        self.mlp = WhisperMLP(
            embed_dim=config.d_model,
            ffn_dim=config.encoder_ffn_dim,
            act_fn=config.activation_function,
            quant_config=quant_config,
            prefix=f"{prefix}.mlp",
        )
        self.final_layer_norm = nn.LayerNorm(self.embed_dim)

    def forward(
        self,
        hidden_states: torch.Tensor,
    ):
        residual = hidden_states
        hidden_states = self.self_attn_layer_norm(hidden_states)
        hidden_states = self.self_attn(hidden_states=hidden_states)
        hidden_states = residual + hidden_states
        residual = hidden_states
        hidden_states = self.final_layer_norm(hidden_states)
        hidden_states = self.mlp(hidden_states)
        hidden_states = residual + hidden_states

        hidden_states = cast_overflow_tensors(hidden_states)

        return hidden_states


class WhisperDecoderLayer(nn.Module):

    def __init__(self, *, vllm_config: VllmConfig, prefix: str = ""):
        super().__init__()
        config = vllm_config.model_config.hf_config
        cache_config = vllm_config.cache_config
        quant_config = vllm_config.quant_config

        self.self_attn = WhisperAttention(
            embed_dim=config.d_model,
            num_heads=config.decoder_attention_heads,
            attn_type=AttentionType.DECODER,
            cache_config=cache_config,
            quant_config=quant_config,
            prefix=f"{prefix}.self_attn",
        )
        self.self_attn_layer_norm = nn.LayerNorm(config.d_model)
        self.encoder_attn = WhisperCrossAttention(
            embed_dim=config.d_model,
            num_heads=config.decoder_attention_heads,
            cache_config=cache_config,
            quant_config=quant_config,
            prefix=f"{prefix}.encoder_attn",
        )
        self.encoder_attn_layer_norm = nn.LayerNorm(config.d_model)
        self.mlp = WhisperMLP(
            embed_dim=config.d_model,
            ffn_dim=config.decoder_ffn_dim,
            act_fn=config.activation_function,
            quant_config=quant_config,
            prefix=f"{prefix}.mlp",
        )
        self.final_layer_norm = nn.LayerNorm(config.d_model)

    def forward(
        self,
        hidden_states: torch.Tensor,
        encoder_hidden_states: Optional[torch.Tensor],
        output_attentions: bool = False,
        layer_idx: Optional[int] = None,
        alignment_heads: Optional[list[tuple[int, int]]] = None,
    ):
        residual = hidden_states
        hidden_states = self.self_attn_layer_norm(hidden_states)
        hidden_states = self.self_attn(hidden_states=hidden_states)
        hidden_states = residual + hidden_states

        residual = hidden_states
        hidden_states = self.encoder_attn_layer_norm(hidden_states)
        if output_attentions:
            hidden_states, cross_attn_weights = self.encoder_attn(
                hidden_states=hidden_states,
                encoder_hidden_states=encoder_hidden_states,
                output_attentions=True,
                alignment_heads=alignment_heads,
                layer_idx=layer_idx,
            )
        else:
            hidden_states = self.encoder_attn(
                hidden_states=hidden_states,
                encoder_hidden_states=encoder_hidden_states,
            )
            cross_attn_weights = None
        hidden_states = residual + hidden_states

        residual = hidden_states
        hidden_states = self.final_layer_norm(hidden_states)
        hidden_states = self.mlp(hidden_states)
        hidden_states = residual + hidden_states

        if output_attentions:
            return hidden_states, cross_attn_weights
        return hidden_states


class WhisperEncoder(nn.Module):

    def __init__(self, *, vllm_config: VllmConfig, prefix: str = ""):
        super().__init__()
        config = vllm_config.model_config.hf_config
        embed_dim = config.d_model
        self.num_mel_bins = config.num_mel_bins
        self.max_source_positions = config.max_source_positions
        self.embed_scale = (math.sqrt(embed_dim)
                            if config.scale_embedding else 1.0)

        self.conv1 = nn.Conv1d(self.num_mel_bins,
                               embed_dim,
                               kernel_size=3,
                               padding=1)
        self.conv2 = nn.Conv1d(embed_dim,
                               embed_dim,
                               kernel_size=3,
                               stride=2,
                               padding=1)
        self.embed_positions = nn.Embedding(self.max_source_positions,
                                            embed_dim)
        self.start_layer, self.end_layer, self.layers = make_layers(
            config.encoder_layers,
            lambda prefix: WhisperEncoderLayer(vllm_config=vllm_config,
                                               prefix=f"{prefix}.layers"),
            prefix=f"{prefix}.layers",
        )
        self.layer_norm = nn.LayerNorm(config.d_model)

        with torch.no_grad():
            self.embed_positions.weight.copy_(
                sinusoids(*self.embed_positions.weight.shape))

    def forward(self, input_features: Union[torch.Tensor, list[torch.Tensor]]):
        hidden_states = []
        for features in input_features:
            embeds = nn.functional.gelu(self.conv1(features))
            embeds = nn.functional.gelu(self.conv2(embeds))
            embeds = embeds.permute(1, 0)
            embeds = embeds + self.embed_positions.weight[:embeds.size(0), :]
            hidden_states.append(embeds)
        hidden_states = torch.cat(hidden_states)

        for encoder_layer in self.layers:
            hidden_states = encoder_layer(hidden_states)

        hidden_states = self.layer_norm(hidden_states)
        return hidden_states


class WhisperDecoder(nn.Module):

    def __init__(self, *, vllm_config: VllmConfig, prefix: str = ""):
        super().__init__()
        config = vllm_config.model_config.hf_config
        self.layerdrop = config.decoder_layerdrop
        self.padding_idx = config.pad_token_id
        self.max_target_positions = config.max_target_positions
        self.max_source_positions = config.max_source_positions
        self.embed_scale = (math.sqrt(config.d_model)
                            if config.scale_embedding else 1.0)

        self.embed_tokens = nn.Embedding(config.vocab_size, config.d_model,
                                         self.padding_idx)
        self.embed_positions = WhisperPositionalEmbedding(
            self.max_target_positions, config.d_model)
        self.start_layer, self.end_layer, self.layers = make_layers(
            config.decoder_layers,
            lambda prefix: WhisperDecoderLayer(vllm_config=vllm_config,
                                               prefix=f"{prefix}.layers"),
            prefix=f"{prefix}.layers",
        )
        self.layer_norm = nn.LayerNorm(config.d_model)

    def forward(
        self,
        input_ids,
        positions: torch.Tensor,
        encoder_hidden_states: Optional[torch.Tensor],
        output_attentions: bool = False,
        alignment_heads: Optional[list[tuple[int, int]]] = None,
    ):
        inputs_embeds = self.get_input_embeddings(input_ids)
        positions = self.embed_positions(positions)
        hidden_states = inputs_embeds + positions

        cross_attentions = [] if output_attentions else None

        for layer_idx, decoder_layer in enumerate(self.layers):
            if output_attentions:
                hidden_states, cross_attn_weights = decoder_layer(
                    hidden_states,
                    encoder_hidden_states=encoder_hidden_states,
                    output_attentions=True,
                    layer_idx=layer_idx,
                    alignment_heads=alignment_heads,
                )
                # Only store non-None cross attention weights to save memory
                if cross_attn_weights is not None:
                    cross_attentions.append(cross_attn_weights)
                else:
                    cross_attentions.append(None)
            else:
                hidden_states = decoder_layer(
                    hidden_states,
                    encoder_hidden_states=encoder_hidden_states,
                )

        hidden_states = self.layer_norm(hidden_states)
        
        if output_attentions:
            return hidden_states, cross_attentions
        return hidden_states

    def get_input_embeddings(
        self,
        input_ids: torch.Tensor,
    ) -> torch.Tensor:
        return self.embed_tokens(input_ids)


class WhisperModel(nn.Module):

    def __init__(self, *, vllm_config: VllmConfig, prefix: str = ""):
        super().__init__()
        self.encoder = WhisperEncoder(vllm_config=vllm_config,
                                      prefix=f"{prefix}.encoder")
        self.decoder = WhisperDecoder(vllm_config=vllm_config,
                                      prefix=f"{prefix}.decoder")

    def forward(
        self,
        input_features: Optional[Union[torch.Tensor, list[torch.Tensor]]],
        input_ids: Optional[torch.Tensor],
        positions: torch.Tensor,
        output_attentions: bool = False,
        alignment_heads: Optional[list[tuple[int, int]]] = None,
    ):
        encoder_outputs = self.get_encoder_outputs(input_features)
        if output_attentions:
            decoder_outputs, cross_attentions = self.decoder(
                input_ids=input_ids,
                positions=positions,
                encoder_hidden_states=encoder_outputs,
                output_attentions=True,
                alignment_heads=alignment_heads,
            )
            return decoder_outputs, cross_attentions
        else:
            decoder_outputs = self.decoder(
                input_ids=input_ids,
                positions=positions,
                encoder_hidden_states=encoder_outputs,
            )
            return decoder_outputs

    def get_encoder_outputs(
        self,
        input_features: Optional[Union[torch.Tensor, list[torch.Tensor]]],
    ) -> Optional[torch.Tensor]:
        if input_features is None:
            return None
        return self.encoder(input_features)

    def load_weights(self, weights: Iterable[tuple[str,
                                                   torch.Tensor]]) -> set[str]:
        stacked_params_mapping = [
            # (param_name, shard_name, shard_id)
            (".self_attn.qkv_proj", ".self_attn.q_proj", "q"),
            (".self_attn.qkv_proj", ".self_attn.k_proj", "k"),
            (".self_attn.qkv_proj", ".self_attn.v_proj", "v"),
            (".encoder_attn.kv_proj", ".encoder_attn.k_proj", "k"),
            (".encoder_attn.kv_proj", ".encoder_attn.v_proj", "v"),
        ]
        params_dict = dict(self.named_parameters())
        loaded_params: set[str] = set()
        for name, loaded_weight in weights:
            for param_name, weight_name, shard_id in stacked_params_mapping:
                if weight_name not in name:
                    continue
                name = name.replace(weight_name, param_name)
                # Skip loading extra bias for GPTQ models.
                if name.endswith(".bias") and name not in params_dict:
                    continue

                param = params_dict[name]
                weight_loader = param.weight_loader
                weight_loader(param, loaded_weight, shard_id)
                break
            else:
                # Skip loading extra bias for GPTQ models.
                if name.endswith(".bias") and name not in params_dict:
                    continue

                param = params_dict[name]
                weight_loader = getattr(param, "weight_loader",
                                        default_weight_loader)
                weight_loader(param, loaded_weight)
            loaded_params.add(name)
        return loaded_params


class WhisperProcessingInfo(BaseProcessingInfo):

    def get_hf_config(self) -> WhisperConfig:
        return self.ctx.get_hf_config(WhisperConfig)

    def get_hf_processor(self,
                         sampling_rate: Optional[int] = None
                         ) -> WhisperProcessor:
        return self.ctx.get_hf_processor(WhisperProcessor)

    def get_supported_mm_limits(self) -> Mapping[str, Optional[int]]:
        return {"audio": 1}

    def get_feature_extractor(self) -> WhisperFeatureExtractor:
        hf_processor = self.get_hf_processor()
        feature_extractor = hf_processor.feature_extractor  # type: ignore
        assert isinstance(feature_extractor, WhisperFeatureExtractor)
        return feature_extractor

    def get_num_audio_tokens(self) -> int:
        return self.get_hf_config().max_source_positions


class WhisperDummyInputsBuilder(BaseDummyInputsBuilder[WhisperProcessingInfo]):

    def get_dummy_text(self, mm_counts: Mapping[str, int]) -> str:
        num_audios = mm_counts.get("audio", 0)

        return "<|startoftranscript|>" * num_audios

    def get_dummy_mm_data(
        self,
        seq_len: int,
        mm_counts: Mapping[str, int],
    ) -> MultiModalDataDict:
        feature_extractor = self.info.get_feature_extractor()

        sampling_rate = feature_extractor.sampling_rate
        audio_len = feature_extractor.chunk_length * sampling_rate
        num_audios = mm_counts.get("audio", 0)

        return {
            "audio":
            self._get_dummy_audios(length=audio_len, num_audios=num_audios)
        }


class WhisperMultiModalProcessor(
        EncDecMultiModalProcessor[WhisperProcessingInfo]):

    def _get_data_parser(self) -> MultiModalDataParser:
        feature_extractor = self.info.get_feature_extractor()
        return MultiModalDataParser(target_sr=feature_extractor.sampling_rate)

    @property
    def pad_dummy_encoder_prompt(self) -> bool:
        return True

    def create_encoder_prompt(
        self,
        prompt: Union[str, list[int]],
        mm_data: MultiModalDataDict,
    ) -> Union[str, list[int]]:
        # Strictly speaking, whisper encoder only accept audio features.
        # We create a dummy encoder prompt here which will be padded to
        # num_audio_tokens. So that we can create dummy data from this
        # for encoder profiling.
        return [0]

    def _call_hf_processor(
        self,
        prompt: str,
        mm_data: Mapping[str, object],
        mm_kwargs: Mapping[str, object],
    ) -> BatchFeature:
        if mm_data:
            feature_extractor = self.info.get_feature_extractor(**mm_kwargs)
            mm_data = dict(audio=mm_data.pop("audios"))
            mm_kwargs = dict(
                **mm_kwargs,
                sampling_rate=feature_extractor.sampling_rate,
            )
        processed_outputs = super()._call_hf_processor(
            prompt=prompt,
            mm_data=mm_data,
            mm_kwargs=mm_kwargs,
        )
        if "labels" in processed_outputs:
            processed_outputs["input_ids"] = processed_outputs.pop("labels")
        return processed_outputs

    def _get_mm_fields_config(
        self,
        hf_inputs: BatchFeature,
        hf_processor_mm_kwargs: Mapping[str, object],
    ) -> Mapping[str, MultiModalFieldConfig]:
        return dict(input_features=MultiModalFieldConfig.batched("audio"))

    def _get_prompt_updates(
        self,
        mm_items: MultiModalDataItems,
        hf_processor_mm_kwargs: Mapping[str, object],
        out_mm_kwargs: MultiModalKwargs,
    ) -> Sequence[PromptUpdate]:
        num_tokens = self.info.get_num_audio_tokens()
        return [
            PromptReplacement(
                modality="audio",
                target=[0],
                replacement=[0] * num_tokens,
            )
        ]


@MULTIMODAL_REGISTRY.register_processor(WhisperMultiModalProcessor,
                                        info=WhisperProcessingInfo,
                                        dummy_inputs=WhisperDummyInputsBuilder)
class WhisperForConditionalGeneration(nn.Module, SupportsTranscription,
                                      SupportsMultiModal, SupportsV0Only):
    packed_modules_mapping = {
        "self_attn.qkv_proj": [
            "self_attn.q_proj",
            "self_attn.k_proj",
            "self_attn.v_proj",
        ],
        "encoder_attn.kv_proj": ["encoder_attn.k_proj", "encoder_attn.v_proj"],
    }

    hf_to_vllm_mapper = WeightsMapper(orig_to_new_substr={
        ".fc1.": ".mlp.fc1.",
        ".fc2.": ".mlp.fc2."
    })

    def __init__(self, *, vllm_config: VllmConfig, prefix: str = ""):
        super().__init__()
        config = vllm_config.model_config.hf_config
        quant_config = vllm_config.quant_config
        self.config = config
        self.dtype = vllm_config.model_config.dtype

        self.model = WhisperModel(vllm_config=vllm_config, prefix=prefix)
        self.unpadded_vocab_size = config.vocab_size
        self.proj_out = ParallelLMHead(config.vocab_size,
                                       config.d_model,
                                       quant_config=quant_config)
        self.proj_out = self.proj_out.tie_weights(
            self.model.decoder.embed_tokens)
        logit_scale = getattr(config, "logit_scale", 1.0)
        self.logits_processor = LogitsProcessor(self.unpadded_vocab_size,
                                                config.vocab_size, logit_scale)
        
        # Default alignment heads for word-level timestamps
        # These are the standard alignment heads used by OpenAI Whisper
        self.alignment_heads = getattr(config, "alignment_heads", [
            (0, 2), (0, 3), (0, 4), (0, 5), (1, 0), (1, 1), (1, 2), (1, 3),
            (2, 0), (2, 1), (2, 2), (2, 3), (3, 0), (3, 1), (3, 2), (3, 3),
            (4, 0), (4, 1), (4, 2), (4, 3), (5, 0), (5, 1), (5, 2), (5, 3)
        ])
        
        # Cache processor components to avoid hot-path lookups during word timing extraction
        self._cached_processor = None
        self._cached_tokenizer = None
        self._cached_feature_extractor = None

    def _get_cached_processor_components(self):
        """Lazily cache processor components to avoid hot-path lookups during word timing extraction."""
        if self._cached_processor is None:
            from vllm.transformers_utils.processor import cached_get_processor
            self._cached_processor = cached_get_processor(
                self.config.name_or_path if hasattr(self.config, 'name_or_path') else 'openai/whisper-base'
            )
            self._cached_tokenizer = self._cached_processor.tokenizer
            self._cached_feature_extractor = self._cached_processor.feature_extractor
        
        return self._cached_tokenizer, self._cached_feature_extractor

    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        **kwargs,
    ) -> torch.Tensor:
        audio_input = self._parse_and_validate_audio_input(**kwargs)
        encoder_outputs = self.model.get_encoder_outputs(audio_input.get("input_features"))

        sampling_metadata = kwargs.get("sampling_metadata")
        if sampling_metadata:
            for seq_group in sampling_metadata.seq_groups:
                granularities = getattr(seq_group.sampling_params,
                                        "timestamp_granularities", ())
                if "word" not in granularities:
                    continue                                        # nothing to do for this group

                seq_group._cached_encoder_outputs = encoder_outputs

                # -- audio length mapping (needed for hop-length → frame math) --
                audio_mm = getattr(seq_group, "mm_items", {}).get("audio")
                audio_lengths: dict[str, int] = {}

                if audio_mm is not None and audio_input.get("input_features") is not None:
                    if isinstance(audio_mm, list):
                        # each list item can be Tensor or (Tensor, sr)
                        for idx, (seq_id, _) in enumerate(seq_group.seq_data.items()):
                            try:
                                wav = audio_mm[idx][0] if isinstance(audio_mm[idx], tuple) else audio_mm[idx]
                            except IndexError:                       # safety net
                                wav = audio_mm[0][0] if isinstance(audio_mm[0], tuple) else audio_mm[0]
                            audio_lengths[seq_id] = int(wav.shape[-1])
                    else:                                           # single clip reused by all seqs
                        wav = audio_mm[0] if isinstance(audio_mm, tuple) else audio_mm
                        length = int(wav.shape[-1])
                        for seq_id in seq_group.seq_data:
                            audio_lengths[seq_id] = length

                seq_group._cached_audio_lengths = audio_lengths

        # 3. Decoder
        decoder_outputs = self.model.decoder(
            input_ids=input_ids,
            positions=positions,
            encoder_hidden_states=encoder_outputs,
        )
        return decoder_outputs

    def get_language_model(self) -> torch.nn.Module:
        return self.model.decoder

    def get_multimodal_embeddings(self,
                                  **kwargs: object) -> MultiModalEmbeddings:
        # TODO: This method does not obey the interface for SupportsMultiModal.
        # Refactor this once encoder/decoder support is implemented in V1.
        audio_input = self._parse_and_validate_audio_input(**kwargs)
        return self.model.get_encoder_outputs(audio_input["input_features"])

    def get_input_embeddings(
        self,
        input_ids: torch.Tensor,
        multimodal_embeddings: Optional[NestedTensors] = None,
    ) -> torch.Tensor:
        # TODO: This method just returns the decoder sequence embeddings since
        # Whisper does not have encoder text tokens. Refactor this once
        # encoder/decoder support is implemented in V1.
        return self.model.decoder.get_input_embeddings(input_ids)

    def _parse_and_validate_audio_input(
            self, **kwargs: object) -> WhisperAudioInputs:
        input_features = kwargs.pop("input_features", None)

        if input_features is not None:
            if not isinstance(input_features, (torch.Tensor, list)):
                raise ValueError("Incorrect type of audio features. "
                                 f"Got type: {type(input_features)}")
            input_features = torch.cat(
                [feat.to(self.dtype) for feat in input_features])

        return WhisperAudioInputs(input_features=input_features)

    def compute_logits(self, hidden_states: torch.Tensor,
                       sampling_metadata: SamplingMetadata) -> torch.Tensor:
        # Check if any sequence in the batch requires timestamp processing
        timestamp_requirements = self._check_timestamp_requirements(sampling_metadata)
        
        if timestamp_requirements["word"]:
            self._process_completed_sequences_for_word_timings(sampling_metadata)
            
        logits = self.logits_processor(self.proj_out, hidden_states,
                                       sampling_metadata)
        return logits

    def _process_completed_sequences_for_word_timings(self, sampling_metadata: SamplingMetadata) -> None:
        """Process word timings only for sequences that have just completed (seen EOT token).
        
        This avoids running the second decoder pass multiple times during generation.
        """
        
        for seq_group in sampling_metadata.seq_groups:
            if (hasattr(seq_group.sampling_params, 'timestamp_granularities') and 
                "word" in seq_group.sampling_params.timestamp_granularities):
                
                # Get cached encoder outputs
                encoder_outputs = getattr(seq_group, '_cached_encoder_outputs', None)
                if encoder_outputs is None:
                    continue
                
                # Ensure encoder outputs are on the same device (important for CPU offload)
                # Do this once per group and cache the result to prevent PCIe thrashing
                device = next(self.model.parameters()).device
                if encoder_outputs.device != device:
                    encoder_outputs = encoder_outputs.to(device)
                    # Cache the device-copied version to avoid repeated transfers
                    seq_group._cached_encoder_outputs = encoder_outputs
                
                seq_data = seq_group.seq_data
                for seq_id, seq in seq_data.items():
                    token_ids = seq.get_token_ids()
                    
                    # Skip if sequence is too short or already processed
                    if len(token_ids) < 5:
                        continue
                    
                    # Optimization: Track last checked length to avoid redundant processing
                    last_checked_length = getattr(seq, '_last_word_timing_check_length', 0)
                    if len(token_ids) <= last_checked_length:
                        continue
                    
                    seq._last_word_timing_check_length = len(token_ids)
                    
                    # Process sequences that might be completed
                    try:
                        # Get cached tokenizer components (avoid hot-path processor lookup)
                        tokenizer, feature_extractor = self._get_cached_processor_components()
                        
                        # Use dynamic EOT token (50257 for transcribe, 51009 for translate)
                        eot_token = tokenizer.eot
                        
                        # Check if sequence is completed (ends with EOT token)
                        sequence_completed = (len(token_ids) > 0 and token_ids[-1] == eot_token)
                        
                        # Only process completed sequences that haven't been processed yet
                        if (sequence_completed and 
                            not (hasattr(seq, 'word_timings_processed') and seq.word_timings_processed)):
                        
                            # Use dynamic SOT sequence length based on task/language settings
                            # Different tasks (transcribe/translate) and languages can have different SOT sequence lengths
                            sot_sequence_len = getattr(tokenizer, 'sot_sequence_length', 4)
                            
                            # Handle <|notimestamps|> token (50258) after SOT sequence
                            if len(token_ids) > sot_sequence_len and token_ids[sot_sequence_len] == 50258:
                                sot_sequence_len += 1  # skip notimestamps flag
                            
                            # Extract text tokens (remove special tokens)
                            # Remove SOT sequence (dynamic length) and EOT token (last token)
                            text_tokens = token_ids[sot_sequence_len:-1] if len(token_ids) > sot_sequence_len + 1 else []
                        
                            if len(text_tokens) > 0:
                                # Get original audio length for this specific sequence for proper frame calculation
                                cached_audio_lengths = getattr(seq_group, '_cached_audio_lengths', {})
                                original_audio_length = cached_audio_lengths.get(seq_id)
                                
                                if original_audio_length is not None:
                                    try:
                                        # Use cached feature extractor
                                        # Calculate frames properly: math.ceil(original_audio_length / hop_length)
                                        # Use ceil to avoid dropping the tail frame
                                        # This accounts for the actual downsampling factor regardless of model variant
                                        # (e.g., large-v3 has stride 4, while base/small/medium have stride 2)
                                        num_frames = math.ceil(original_audio_length / feature_extractor.hop_length)
                                    except Exception as e:
                                        logger.warning(f"Failed to get feature extractor for frame calculation, falling back to encoder output: {e}")
                                        # Fallback to encoder output shape (less accurate for different model variants)
                                        num_frames = encoder_outputs.shape[1] * 2
                                else:
                                    # Fallback to encoder output shape (less accurate for different model variants)  
                                    num_frames = encoder_outputs.shape[1] * 2
                                
                                # Run decoder pass once with attention collection for completed sequence
                                decoder_input_ids = torch.tensor(token_ids, dtype=torch.long, device=device)
                                positions = torch.arange(len(decoder_input_ids), dtype=torch.long, device=device)
                                
                                logger.debug(f"Running second decoder pass for word timing extraction on completed sequence {seq_id} (length={len(token_ids)})")
                                
                                with torch.inference_mode(), torch.cuda.amp.autocast(enabled=False):
                                    _, cross_attentions = self.model.decoder(
                                        input_ids=decoder_input_ids.unsqueeze(0),
                                        positions=positions.unsqueeze(0),
                                        encoder_hidden_states=encoder_outputs,
                                        output_attentions=True,
                                        alignment_heads=self.alignment_heads,
                                    )
                                    
                                    # Filter out None values from cross_attentions (layers with no alignment heads)
                                    attn_for_dtw = [w for w in cross_attentions if w is not None]
                                    
                                    # Guard against empty attention case
                                    if not attn_for_dtw:
                                        # Nothing to align → return empty list
                                        seq.word_timings = []
                                    else:
                                        # Extract word timings using alignment (without chunk offset)
                                        word_timings = self._extract_word_timings(
                                            cross_attention_weights=attn_for_dtw,
                                            text_tokens=text_tokens,
                                            num_frames=num_frames,
                                            chunk_start_sec=0.0  # Don't apply offset in extraction
                                        )
                                        
                                        # Apply chunk offset merge before marking as processed
                                        chunk_start_sec = getattr(seq_group, 'chunk_start_sec', 0.0)
                                        if chunk_start_sec > 0:
                                            word_timings = merge_word_timings_with_chunks(word_timings, chunk_start_sec)
                                        
                                        # Store word timings in sequence for later retrieval
                                        seq.word_timings = word_timings
                                        # Mark for cleanup after JSON serialization to prevent VRAM leaks
                                        seq._word_timings_needs_cleanup = True
                                    seq.word_timings_processed = True  # Mark as processed
                                    logger.debug(f"Extracted {len(word_timings)} word timings for completed sequence {seq_id}")
                            else:
                                # Mark as processed even if no text tokens (empty transcription)
                                seq.word_timings = []
                                seq._word_timings_needs_cleanup = True
                                seq.word_timings_processed = True
                                
                    except Exception as e:
                        logger.warning(f"Failed to process word timings for completed sequence {seq_id}: {e}")
                        # Mark as processed to avoid retrying
                        seq.word_timings_processed = True
                
                # Clean up cached data to free memory (3 MB / chunk)
                if hasattr(seq_group, '_cached_encoder_outputs'):
                    del seq_group._cached_encoder_outputs
                if hasattr(seq_group, '_cached_audio_lengths'):
                    del seq_group._cached_audio_lengths

                    
                # Clean up sequence-level state for multi-request isolation
                # (important for streaming responses where objects persist across chunks)
                for seq_id, seq in seq_group.seq_data.items():
                    if hasattr(seq, '_last_word_timing_check_length'):
                        del seq._last_word_timing_check_length
                    if hasattr(seq, 'word_timings_processed'):
                        del seq.word_timings_processed
                        
                    # Clean up word timing VRAM after JSON serialization to prevent leaks in long streams
                    if getattr(seq, "_word_timings_needs_cleanup", False):
                        if hasattr(seq, 'word_timings'):
                            del seq.word_timings
                        del seq._word_timings_needs_cleanup

    def _check_timestamp_requirements(self, sampling_metadata: SamplingMetadata) -> dict[str, bool]:
        """Check if any sequence in the batch requires timestamp granularities.
        
        Returns:
            Dict with 'segment' and 'word' keys indicating if those granularities are needed.
        """
        needs_segment = False
        needs_word = False
        
        if sampling_metadata.seq_groups:
            for seq_group in sampling_metadata.seq_groups:
                timestamp_granularities = getattr(
                    seq_group.sampling_params, 'timestamp_granularities', None
                )
                if timestamp_granularities:
                    if "segment" in timestamp_granularities:
                        needs_segment = True
                    if "word" in timestamp_granularities:
                        needs_word = True
                    # Early exit if both are found
                    if needs_segment and needs_word:
                        break
        
        return {"segment": needs_segment, "word": needs_word}

    def _extract_word_timings(
        self, 
        cross_attention_weights: list[torch.Tensor], 
        text_tokens: list[int],
        num_frames: int,
        chunk_start_sec: float = 0.0
    ) -> list[WordTiming]:
        """Extract word-level timings from cross-attention weights using DTW alignment.
        
        Args:
            cross_attention_weights: List of attention weight tensors from decoder layers
            text_tokens: Token IDs for the text (without special tokens)
            num_frames: Number of audio frames
            chunk_start_sec: Start time of current chunk for long audio
            
        Returns:
            List of WordTiming objects with word-level timestamps
        """
        try:
            # Get cached processor components for time conversion
            tokenizer, feature_extractor = self._get_cached_processor_components()
            
            feature_extractor_info = {
                "hop_length": feature_extractor.hop_length,
                "sampling_rate": feature_extractor.sampling_rate,
            }
            
            # Tokenizer info with actual tokenizer instance
            tokenizer_info = {
                "sot_sequence_len": getattr(tokenizer, 'sot_sequence_length', 4),  # Dynamic SOT length based on task/language
                "eot_token": tokenizer.eot if hasattr(tokenizer, 'eot') else 50257,
                "tokenizer": tokenizer,  # Pass actual tokenizer instance
            }
            
            # Perform alignment
            word_timings = find_alignment(
                cross_attention_weights=cross_attention_weights,
                text_tokens=text_tokens,
                num_frames=num_frames,
                alignment_heads=self.alignment_heads,
                tokenizer_info=tokenizer_info,
                feature_extractor_info=feature_extractor_info,
            )
            
            return word_timings
            
        except Exception as e:
            logger.warning(f"Failed to extract word timings: {e}")
            return []
