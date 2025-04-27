# LayerSkip Project: Optimizing Small Language Models

## Abstract

Large Language Models (LLMs) offer state-of-the-art performance across many NLP tasks but suffer from high computational costs due to their deep transformer architectures. To address this, we investigate the use of early-exit strategies to reduce inference time by dynamically halting token processing when confidence thresholds are met. Our work builds upon the LayerSkip framework, incorporating layer dropout and exploring Self Speculative Decoding (SSD) and Dynamic Early Exit (DEE) techniques-specifically. Our work focuses on smaller decoder-only transformer models, to lead an investigation into the possible speedup for models with fewer layers available to skip. We train and evaluate small Transformer models on the WikiText-2 dataset, investigating speedups for low parameter models. We evaluate DEE and SSD applied independently during inference—without requiring changes to the model's training procedure. Using the WikiText-2 dataset, we assess both speedups and generation quality, observing meaningful reductions in compute while highlighting challenges in preserving output fidelity. This evaluation demonstrates the potential for post-training inference-time optimizations in lightweight LLMs.

## Project Structure

```
.
├── src/                    # Main source code
│   ├── train_decoder.py    # Training script
│   ├── generate_decoder_text.py  # Text generation script
│   ├── decoder_model.py    # Model architecture
│   ├── decoder_checkpoints/ # Saved model checkpoints
│   └── LayerSkip Modifications/  # Early exit and speculative decoding
│       ├── infer.py        # Inference with DEE and SSD
│       ├── model.py        # Model architecture
│       ├── decoder.py      # Decoder implementation
│       ├── calibrate.py    # Threshold calibration
│       ├── calm_thresholds.json  # CALM thresholds
│       └── mymodel_earlyexit.pt  # Pre-trained model
└── logs/                   # Training and evaluation logs
```

## Setup

1. Create a Python virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

2. Install required packages:
```bash
pip install torch torchvision torchaudio
pip install transformers datasets sentencepiece
pip install numpy tqdm
```

## Training a Model

To train a new model, use the `train_decoder.py` script:

```bash
python src/train_decoder.py --train \
    --d_model 128 \
    --num_layers 6 \
    --num_heads 8 \
    --d_ff 512 \
    --dropout 0.1 \
    --max_skip_prob 0.1 \
    --epochs 5 \
    --batch_size 64 \
    --lr 1e-4
```

Key parameters:
- `d_model`: Embedding dimension (default: 128)
- `num_layers`: Number of transformer layers (default: 6)
- `num_heads`: Number of attention heads (default: 8)
- `d_ff`: Feed-forward dimension (default: 512)
- `dropout`: Dropout rate (default: 0.1)
- `max_skip_prob`: Maximum layer dropout probability (default: 0.1)
- `epochs`: Number of training epochs (default: 5)
- `batch_size`: Training batch size (default: 64)
- `lr`: Learning rate (default: 1e-4)

## Generating Text

To generate text using a trained model:

```bash
python src/generate_decoder_text.py \
    --checkpoint path/to/checkpoint.pt \
    --prompt "Your prompt here" \
    --max_length 100 \
    --temperature 1.0
```

Parameters:
- `checkpoint`: Path to model checkpoint
- `prompt`: Initial text prompt
- `max_length`: Maximum length of generated text
- `temperature`: Sampling temperature (higher = more random)

## Example Usage

Here are some quick examples to get started with the src code:

1. Train a small model:
```bash
# Train a 6-layer model with default parameters
python src/train_decoder.py --train \
    --d_model 128 \
    --num_layers 6 \
    --num_heads 8 \
    --d_ff 512 \
    --dropout 0.1 \
    --max_skip_prob 0.1 \
    --epochs 5
```

2. Generate text using a trained model:
```bash
# Generate text with default parameters
python src/generate_decoder_text.py \
    --checkpoint decoder_checkpoints/best_decoder_d128_h8_l6_ff512_dp0.1_ls0_e1_val6.4181.pt \
    --prompt "The history of language models" \
    --max_length 50

# Generate more creative text with higher temperature
python src/generate_decoder_text.py \
    --checkpoint decoder_checkpoints/best_decoder_d128_h8_l6_ff512_dp0.1_ls0_e1_val6.4181.pt \
    --prompt "Once upon a time" \
    --max_length 100 \
    --temperature 1.5
```

3. Evaluate a trained model:
```bash
# Evaluate model performance
python src/train_decoder.py --eval \
    --checkpoint decoder_checkpoints/best_decoder_d128_h8_l6_ff512_dp0.1_ls0_e1_val6.4181.pt
```

4. Train and evaluate in one command:
```bash
# Train for 5 epochs and evaluate
python src/train_decoder.py --train --eval \
    --d_model 128 \
    --num_layers 6 \
    --num_heads 8 \
    --d_ff 512 \
    --dropout 0.1 \
    --max_skip_prob 0.1 \
    --epochs 5
```

The model checkpoints will be saved in the `decoder_checkpoints` directory with names indicating their configuration and performance.

## LayerSkip Modifications

The `src/LayerSkip Modifications` directory contains the implementation of early exit and speculative decoding techniques:

### Directory Structure
```
LayerSkip Modifications/
├── infer.py              # Inference with DEE and SSD
├── model.py              # Model architecture
├── decoder.py            # Decoder implementation
├── calibrate.py          # Threshold calibration
├── calm_thresholds.json  # CALM thresholds for DEE
├── mymodel_earlyexit.pt  # Pre-trained model with LayerSkip
└── mymodel_earlyexit_noLS.pt  # Pre-trained model without LayerSkip
```

### Dynamic Early Exit (DEE)
To run inference with Dynamic Early Exit:
```bash
python src/LayerSkip\ Modifications/infer.py \
    --checkpoint mymodel_earlyexit.pt \
    --prompt "Your prompt here" \
    --mode dee \
    --conf 0.9
```

Parameters:
- `--checkpoint`: Path to model checkpoint
- `--prompt`: Input text prompt
- `--mode`: Set to "dee" for Dynamic Early Exit
- `--conf`: Confidence threshold (default: 0.9)
- `--lam_file`: Optional JSON file with per-layer thresholds

### Self-Speculative Decoding (SSD)
To run inference with Self-Speculative Decoding:
```bash
python src/LayerSkip\ Modifications/infer.py \
    --checkpoint mymodel_earlyexit.pt \
    --prompt "Your prompt here" \
    --mode ssd \
    --draft_layer 3 \
    --k 4
```

Parameters:
- `--checkpoint`: Path to model checkpoint
- `--prompt`: Input text prompt
- `--mode`: Set to "ssd" for Self-Speculative Decoding
- `--draft_layer`: Layer to use for draft generation
- `--k`: Number of tokens to draft before verification

### Threshold Calibration
To calibrate thresholds for DEE:
```bash
python src/LayerSkip\ Modifications/calibrate.py \
    --checkpoint mymodel_earlyexit.pt \
    --output calm_thresholds.json
```

This will generate per-layer confidence thresholds optimized for your model.

## Model Architecture

The project implements a decoder-only transformer model with the following key features:
- LayerDropout during training
- Dynamic Early Exit (DEE) for inference
- Self-Speculative Decoding (SSD) for faster generation
- Custom tokenizer based on SentencePiece

## Results and Evaluation

The model is evaluated on the WikiText-2 dataset. Key metrics include:
- Validation loss
- Layer-wise performance
- Generation quality
- Inference speed with and without early exit

## Notes

- The model is trained on WikiText-2 dataset
- Early exit techniques are most effective with larger models
- LayerDropout helps strengthen early layers
- SSD provides speedup by batch-verifying tokens

## Team Members

- Aditya Bajoria
- Jay Javeri
- Siddhant Agarwal
- Shrey Gupta