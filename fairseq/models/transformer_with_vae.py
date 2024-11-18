import math
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
from torch import Tensor

from fairseq.models import FairseqEncoder, register_model, register_model_architecture
from fairseq.modules import (
    FairseqDropout,
    LayerDropModuleList,
    LayerNorm,
    PositionalEmbedding,
    transformer_layer,
)
from fairseq.models.transformer import (
    TransformerDecoderBase,
    TransformerEncoderBase,
    TransformerModel,
    TransformerConfig,
    DEFAULT_MAX_SOURCE_POSITIONS,
    DEFAULT_MAX_TARGET_POSITIONS,
    DEFAULT_MIN_PARAMS_TO_WRAP,
)
from fairseq.models.transformer import base_architecture as transformer_base_architecture
from fairseq.modules.quant_noise import quant_noise as apply_quant_noise_
from fairseq.dataclass.utils import gen_parser_from_dataclass

import logging
logger = logging.getLogger(__name__)

@dataclass 
class TransformerWithVAEConfig(TransformerConfig):
    sample_beta: float = field(
        default=1.0,
        metadata={"help": "weight of kld loss"},
    )
    gen_vae: bool = field(
        default=False,
        metadata={"help": "using vae for generation"},
    )
    
class MLP(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int, output_dim: int, num_layers: int = 2, activation_fn=nn.GELU(), dropout: float = 0.2):
        super().__init__()
        layers = [
            nn.Sequential(nn.Linear(input_dim if i == 0 else hidden_dim, hidden_dim), activation_fn, nn.Dropout(dropout))
            for i in range(num_layers)
        ]
        self.layers = nn.ModuleList(layers)
        self.out_proj = nn.Linear(hidden_dim, output_dim)

    def forward(self, x: Tensor) -> Tensor:
        for layer in self.layers:
            x = layer(x)
        return self.out_proj(x)
    
    
class VAEEncoder(TransformerEncoderBase):
    def __init__(self, cfg, src_dict, tgt_dict, src_embed_tokens, tgt_embed_tokens, return_fc=False):
        super().__init__(cfg, src_dict, src_embed_tokens, return_fc)
        
        self.dim = src_embed_tokens.embedding_dim
        self.src_embed_tokens = src_embed_tokens
        self.tgt_embed_tokens = tgt_embed_tokens
        
        self.embed_scale = 1.0 if cfg.no_scale_embedding else math.sqrt(self.dim)
            
        self.tgt_proj = nn.Linear(cfg.decoder.embed_dim, cfg.encoder.embed_dim) if cfg.decoder.embed_dim != cfg.encoder.embed_dim else None
        self.out_proj = MLP(self.dim, self.dim, 2 * self.dim)
        self.configure_embeddings(cfg)
        
        
    def configure_embeddings(self, cfg):
        self.src_embed_positions = self.build_positional_embedding(cfg, cfg.max_source_positions, self.src_embed_tokens.embedding_dim, learned=cfg.encoder.learned_pos)
        self.tgt_embed_positions = self.build_positional_embedding(cfg, cfg.max_target_positions, self.tgt_embed_tokens.embedding_dim, learned=cfg.decoder.learned_pos)
        self.layernorm_embedding = LayerNorm(self.dim, export=cfg.export) if cfg.layernorm_embedding else None
        self.quant_noise = apply_quant_noise_(
            nn.Linear(self.dim, self.dim, bias=False),
            cfg.quant_noise.pq,
            cfg.quant_noise.pq_block_size,
        ) if cfg.quant_noise.pq > 0 else None

    def build_positional_embedding(self, cfg, max_positions: int, embed_dim: int, learned: bool = True):
        if not cfg.no_token_positional_embeddings:
            return PositionalEmbedding(max_positions, embed_dim, self.padding_idx, learned=learned)
        return None
    
    def forward_embedding(self, src_tokens, tgt_tokens):
        src_embedding = self.src_embed_tokens(src_tokens)
        tgt_embedding = self.tgt_embed_tokens(tgt_tokens)
        x_src = src_embed = self.embed_scale * src_embedding
        x_tgt = tgt_embed = self.embed_scale * tgt_embedding
        
        if self.tgt_proj:
            x_tgt = self.tgt_proj(x_tgt)
        
        if self.src_embed_positions is not None:
            x_src += self.src_embed_positions(src_tokens)
        if self.tgt_embed_positions is not None:
            x_tgt += self.tgt_embed_positions(tgt_tokens)
        if self.layernorm_embedding is not None:
            x_src = self.layernorm_embedding(x_src)
            x_tgt = self.layernorm_embedding(x_tgt)
        x_src = self.dropout_module(x_src)
        x_tgt = self.dropout_module(x_tgt)
        if self.quant_noise is not None:
            x_src = self.quant_noise(x_src)
            x_tgt = self.quant_noise(x_tgt)
        return x_src, x_tgt, src_embed, tgt_embed
    
    def forward(self, src_tokens, tgt_tokens, src_lengths: Optional[torch.Tensor] = None, tgt_lengths: Optional[torch.Tensor] = None, return_all_hiddens: bool = False):
        return self.forward_scriptable(
            src_tokens, tgt_tokens, src_lengths, tgt_lengths, return_all_hiddens
        )
    
    def forward_scriptable(self, src_tokens, tgt_tokens, src_lengths: Optional[torch.Tensor] = None, tgt_lengths: Optional[torch.Tensor] = None, return_all_hiddens: bool = False):
        x_src, x_tgt, src_embed, tgt_embed = self.forward_embedding(src_tokens, tgt_tokens)
        src_mask = src_tokens.eq(self.padding_idx)
        tgt_mask = tgt_tokens.eq(self.padding_idx)
        x = torch.cat((x_src, x_tgt), dim=1)
        encoder_padding_mask = torch.cat((src_mask, tgt_mask), dim=1)
        has_pads = encoder_padding_mask.any()
        # Torchscript doesn't handle bool Tensor correctly, so we need to work around.
        if torch.jit.is_scripting():
            has_pads = torch.tensor(1) if has_pads else torch.tensor(0)
        
        # account for padding while computing the representation
        x = x * (
            1 - encoder_padding_mask.unsqueeze(-1).type_as(x) * has_pads.type_as(x)
        )
        
        # B x T x C -> T x B x C
        x = x.transpose(0, 1)

        encoder_states = []
        fc_results = []

        if return_all_hiddens:
            encoder_states.append(x)
            
        # encoder layers
        for layer in self.layers:
            lr = layer(
                x, encoder_padding_mask=encoder_padding_mask if has_pads else None
            )

            if isinstance(lr, tuple) and len(lr) == 2:
                x, fc_result = lr
            else:
                x = lr
                fc_result = None

            if return_all_hiddens and not torch.jit.is_scripting():
                assert encoder_states is not None
                encoder_states.append(x)
                fc_results.append(fc_result)
        
        # Only the src part is used for the encoder output
        x = x[:src_tokens.size(1)]

        if self.layer_norm is not None:
            x = self.layer_norm(x)
        
        x = self.out_proj(x)
        
        return x
        

class TransformerEncoderWithVAE(FairseqEncoder):
    def __init__(self, cfg, src_dict, tgt_dict, encoder_embed_tokens, decoder_embed_tokens):
        super().__init__(src_dict)
        self.main_encoder = TransformerEncoderBase(cfg, src_dict, encoder_embed_tokens)
        self.vae_encoder = VAEEncoder(cfg, src_dict, tgt_dict, encoder_embed_tokens, decoder_embed_tokens)
        self.sample_beta = cfg.sample_beta
        self.gen_vae = cfg.gen_vae
    
    def forward(self, src_tokens, tgt_tokens, src_lengths, tgt_lengths, return_all_hiddens: bool = False):
        main_encoder_out = self.main_encoder(src_tokens, src_lengths, return_all_hiddens=return_all_hiddens)
        if (self.training or (not self.training and self.gen_vae)) and (tgt_tokens is not None):
            vae_encoder_out = self.vae_encoder(src_tokens, tgt_tokens, src_lengths, tgt_lengths, return_all_hiddens=return_all_hiddens)
            mean, logstd = torch.chunk(vae_encoder_out, chunks=2, dim=-1)
            # mean = torch.mean(mean, dim=-1, keepdim=True)
            # logstd = torch.mean(logstd, dim=-1, keepdim=True)
            if self.training:
                z = mean + torch.exp(logstd) * torch.randn_like(mean)
            else:
                z = mean + torch.exp(logstd) * torch.randn_like(mean) * self.sample_beta
        else:
            z = torch.randn_like(main_encoder_out['encoder_out'][0]) * self.sample_beta
            mean, logstd = None, None
        return {
            "encoder_out": [main_encoder_out['encoder_out'][0] + z],
            "encoder_padding_mask": main_encoder_out['encoder_padding_mask'],
            "encoder_embedding": main_encoder_out['encoder_embedding'],
            "encoder_states": main_encoder_out['encoder_states'],
            "src_tokens": main_encoder_out['src_tokens'],
            "src_lengths": main_encoder_out['src_lengths'],
            "vae_mean": mean,
            "vae_logstd": logstd,
        }
    
    @torch.jit.export
    def reorder_encoder_out(self, encoder_out, new_order):
        """
        Reorder encoder output according to *new_order*.

        Args:
            encoder_out: output from the ``forward()`` method
            new_order (LongTensor): desired order

        Returns:
            *encoder_out* rearranged according to *new_order*
        """
        if len(encoder_out["encoder_out"]) == 0:
            new_encoder_out = []
        else:
            new_encoder_out = [encoder_out["encoder_out"][0].index_select(1, new_order)]
        if len(encoder_out["encoder_padding_mask"]) == 0:
            new_encoder_padding_mask = []
        else:
            new_encoder_padding_mask = [
                encoder_out["encoder_padding_mask"][0].index_select(0, new_order)
            ]
        if len(encoder_out["encoder_embedding"]) == 0:
            new_encoder_embedding = []
        else:
            new_encoder_embedding = [
                encoder_out["encoder_embedding"][0].index_select(0, new_order)
            ]

        if len(encoder_out["src_tokens"]) == 0:
            src_tokens = []
        else:
            src_tokens = [(encoder_out["src_tokens"][0]).index_select(0, new_order)]

        if len(encoder_out["src_lengths"]) == 0:
            src_lengths = []
        else:
            src_lengths = [(encoder_out["src_lengths"][0]).index_select(0, new_order)]

        encoder_states = encoder_out["encoder_states"]
        if len(encoder_states) > 0:
            for idx, state in enumerate(encoder_states):
                encoder_states[idx] = state.index_select(1, new_order)

        try:
            new_vae_mean = encoder_out["vae_mean"].index_select(1, new_order)
            new_vae_logstd = encoder_out["vae_logstd"].index_select(1, new_order)
        except KeyError:
            new_vae_mean = None
            new_vae_logstd = None
        
        return {
            "encoder_out": new_encoder_out,  # T x B x C
            "encoder_padding_mask": new_encoder_padding_mask,  # B x T
            "encoder_embedding": new_encoder_embedding,  # B x T x C
            "encoder_states": encoder_states,  # List[T x B x C]
            "src_tokens": src_tokens,  # B x T
            "src_lengths": src_lengths,  # B x 1
            "vae_mean": new_vae_mean, # T x B x C
            "vae_logstd": new_vae_logstd, # T x B x C
        }

@register_model("transformer_vae")
class TransformerWithVAE(TransformerModel):

    def __init__(self, args, encoder, decoder):
        cfg = TransformerWithVAEConfig.from_namespace(args)
        super().__init__(cfg, encoder, decoder)
        self.args = args
    
    @classmethod
    def add_args(cls, parser):
        """Add model-specific arguments to the parser."""
        # we want to build the args recursively in this case.
        # do not set defaults so that settings defaults from various architectures still works
        gen_parser_from_dataclass(
            parser, TransformerWithVAEConfig(), delete_default=True, with_prefix=""
        )

    @classmethod
    def build_encoder(cls, cfg, src_dict, tgt_dict, src_embed_tokens, tgt_embed_tokens):
        return TransformerEncoderWithVAE(cfg, src_dict, tgt_dict, src_embed_tokens, tgt_embed_tokens)
    
    @classmethod
    def build_model(cls, args, task):
        """Build a new model instance."""

        # make sure all arguments are present in older models
        base_architecture(args)

        if args.encoder_layers_to_keep:
            args.encoder_layers = len(args.encoder_layers_to_keep.split(","))
        if args.decoder_layers_to_keep:
            args.decoder_layers = len(args.decoder_layers_to_keep.split(","))

        if getattr(args, "max_source_positions", None) is None:
            args.max_source_positions = DEFAULT_MAX_SOURCE_POSITIONS
        if getattr(args, "max_target_positions", None) is None:
            args.max_target_positions = DEFAULT_MAX_TARGET_POSITIONS

        src_dict, tgt_dict = task.source_dictionary, task.target_dictionary

        if args.share_all_embeddings:
            if src_dict != tgt_dict:
                raise ValueError("--share-all-embeddings requires a joined dictionary")
            if args.encoder_embed_dim != args.decoder_embed_dim:
                raise ValueError(
                    "--share-all-embeddings requires --encoder-embed-dim to match --decoder-embed-dim"
                )
            if args.decoder_embed_path and (
                args.decoder_embed_path != args.encoder_embed_path
            ):
                raise ValueError(
                    "--share-all-embeddings not compatible with --decoder-embed-path"
                )
            args.share_decoder_input_output_embed = True

        if getattr(args, "offload_activations", False):
            args.checkpoint_activations = True  # offloading implies checkpointing

        if not args.share_all_embeddings:
            args.min_params_to_wrap = getattr(
                args, "min_params_to_wrap", DEFAULT_MIN_PARAMS_TO_WRAP
            )
        cfg = TransformerWithVAEConfig.from_namespace(args)
        
        # --  TODO T96535332
        #  bug caused by interaction between OmegaConf II and argparsing
        cfg.decoder.input_dim = int(cfg.decoder.input_dim)
        cfg.decoder.output_dim = int(cfg.decoder.output_dim)
        # --

        if cfg.encoder.layers_to_keep:
            cfg.encoder.layers = len(cfg.encoder.layers_to_keep.split(","))
        if cfg.decoder.layers_to_keep:
            cfg.decoder.layers = len(cfg.decoder.layers_to_keep.split(","))

        src_dict, tgt_dict = task.source_dictionary, task.target_dictionary

        if cfg.share_all_embeddings:
            if src_dict != tgt_dict:
                raise ValueError("--share-all-embeddings requires a joined dictionary")
            if cfg.encoder.embed_dim != cfg.decoder.embed_dim:
                raise ValueError(
                    "--share-all-embeddings requires --encoder-embed-dim to match --decoder-embed-dim"
                )
            if cfg.decoder.embed_path and (
                cfg.decoder.embed_path != cfg.encoder.embed_path
            ):
                raise ValueError(
                    "--share-all-embeddings not compatible with --decoder-embed-path"
                )
            encoder_embed_tokens = cls.build_embedding(
                cfg, src_dict, cfg.encoder.embed_dim, cfg.encoder.embed_path
            )
            decoder_embed_tokens = encoder_embed_tokens
            cfg.share_decoder_input_output_embed = True
        elif cfg.merge_src_tgt_embed:
            logger.info(f"source dict size: {len(src_dict)}")
            logger.info(f"target dict size: {len(tgt_dict)}")
            src_dict.update(tgt_dict)
            task.src_dict = src_dict
            task.tgt_dict = src_dict
            logger.info(f"merged dict size: {len(src_dict)}")
            encoder_embed_tokens = cls.build_embedding(
                cfg, src_dict, cfg.encoder.embed_dim
            )
            decoder_embed_tokens = encoder_embed_tokens
            cfg.share_decoder_input_output_embed = True
        else:
            encoder_embed_tokens = cls.build_embedding(
                cfg, src_dict, cfg.encoder.embed_dim, cfg.encoder.embed_path
            )
            decoder_embed_tokens = cls.build_embedding(
                cfg, tgt_dict, cfg.decoder.embed_dim, cfg.decoder.embed_path
            )
        if cfg.offload_activations:
            cfg.checkpoint_activations = True  # offloading implies checkpointing
        
        encoder = cls.build_encoder(cfg, src_dict, tgt_dict, encoder_embed_tokens, decoder_embed_tokens)
        decoder = cls.build_decoder(cfg, tgt_dict, decoder_embed_tokens)
        return cls(cfg, encoder, decoder)
    
    def forward(
        self,
        src_tokens,
        tgt_tokens,
        src_lengths,
        tgt_lengths,
        prev_output_tokens,
        return_all_hiddens: bool = True,
        features_only: bool = False,
        alignment_layer: Optional[int] = None,
        alignment_heads: Optional[int] = None,
    ):
        encoder_out = self.encoder(src_tokens, tgt_tokens, src_lengths, tgt_lengths, return_all_hiddens=return_all_hiddens)
        x, extra = self.decoder(
            prev_output_tokens,
            encoder_out=encoder_out,
            features_only=features_only,
            alignment_layer=alignment_layer,
            alignment_heads=alignment_heads,
            src_lengths=src_lengths,
            return_all_hiddens=return_all_hiddens,
        )
        return x, extra, encoder_out["vae_mean"], encoder_out["vae_logstd"], encoder_out["encoder_padding_mask"][0]
    
@register_model_architecture("transformer_vae", "transformer_vae")
def base_architecture(args):
    transformer_base_architecture(args)
    args.sample_beta = getattr(args, "sample_beta", 1.0)
    args.gen_vae = getattr(args, "gen_vae", False)
 