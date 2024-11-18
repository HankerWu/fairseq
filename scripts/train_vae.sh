#!/bin/bash

BLOB_PATH="/blob"
if [ ! -d "$BLOB_PATH" ]; then
BLOB_PATH="/home/v-wukehan/blob1.v2"
fi
    

# Set default values for the parameters
lr=0.0003                          # Reduced learning rate for large datasets
warmup=10000                       # Higher warmup for gradual adjustment
dropout=0.1                        # Lower dropout for large datasets
layers=12                          # Standard depth for transformers
vae_beta=1.0                       # Default beta for VAE term
sample_beta=1.0                    # Default sample beta
embed_dim=768                      # Common embedding size for robust representation
attention_heads=12                 # Suitable for medium-large transformers
max_tokens=16384                   # Increased batch size for large data
weight_decay=0.01                  # Slight weight decay to prevent overfitting
base_savedir="checkpoints/vae"
extra_args=""

echo "Starting training script with initial setup..."

# Parse command-line arguments
while [[ $# -gt 0 ]]; do
    key="$1"
    case $key in
        --lr)
            lr="$2"
            shift; shift
            ;;
        --warmup)
            warmup="$2"
            shift; shift
            ;;
        --dropout)
            dropout="$2"
            shift; shift
            ;;
        --layers)
            layers="$2"
            shift; shift
            ;;
        --vae-beta)
            vae_beta="$2"
            shift; shift
            ;;
        --sample-beta)
            sample_beta="$2"
            shift; shift
            ;;
        --embed-dim)
            embed_dim="$2"
            shift; shift
            ;;
        --attention-heads)
            attention_heads="$2"
            shift; shift
            ;;
        --max-tokens)
            max_tokens="$2"
            shift; shift
            ;;
        --weight-decay)
            weight_decay="$2"
            shift; shift
            ;;
        --base-savedir)
            base_savedir="$2"
            shift; shift
            ;;
        *)
            extra_args+="$1 "
            shift
            ;;
    esac
done

# Create a unique save directory based on parameters
savedir="${base_savedir}/lr${lr}_warmup${warmup}_dropout${dropout}_layers${layers}_vae${vae_beta}_sample${sample_beta}_embed${embed_dim}_attn${attention_heads}_maxtokens${max_tokens}_wd${weight_decay}"
mkdir -p "$savedir"

# Log the parameter settings
echo "Training with the following parameters:"
echo "  Learning Rate: $lr"
echo "  Warmup Steps: $warmup"
echo "  Dropout Rate: $dropout"
echo "  Layers: $layers"
echo "  VAE Beta: $vae_beta"
echo "  Sample Beta: $sample_beta"
echo "  Embedding Dimension: $embed_dim"
echo "  Attention Heads: $attention_heads"
echo "  Max Tokens per Batch: $max_tokens"
echo "  Weight Decay: $weight_decay"
echo "  Base Save Directory: $base_savedir"
echo "  Additional Arguments: $extra_args"

# Run the fairseq-train command with specified parameters
echo "Starting fairseq training..."

set -x
fairseq-train \
    $BLOB_PATH/v-kehanwu/data/msa/bin \
    -s pro -t msa \
    --task translation \
    --ddp-backend no_c10d \
    --arch transformer_vae \
    --share-decoder-input-output-embed \
    --share-all-embeddings \
    --optimizer adam --adam-betas '(0.9, 0.98)' \
    --clip-norm 0.0 \
    --lr-scheduler inverse_sqrt --warmup-updates "$warmup" \
    --warmup-init-lr 1e-07 \
    --max-tokens "$max_tokens" \
    --lr "$lr" \
    --dropout "$dropout" \
    --weight-decay "$weight_decay" \
    --criterion label_smoothed_cross_entropy_with_vae \
    --label-smoothing 0.1 \
    --vae-beta "$vae_beta" \
    --sample-beta "$sample_beta" \
    --encoder-layers "$layers" --decoder-layers "$layers" \
    --encoder-embed-dim "$embed_dim" --decoder-embed-dim "$embed_dim" \
    --encoder-ffn-embed-dim $((embed_dim * 4)) --decoder-ffn-embed-dim $((embed_dim * 4)) \
    --encoder-attention-heads "$attention_heads" --decoder-attention-heads "$attention_heads" \
    --tensorboard-logdir "$savedir/tensorboard_logs" \
    --log-interval 100 \
    --log-format simple \
    --save-dir "$savedir" \
    --keep-last-epochs 3 \
    --fp16 \
    --save-interval-updates 10000 \
    $extra_args

set +x
echo "Training completed. Checkpoints saved in $savedir"
