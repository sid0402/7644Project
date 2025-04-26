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
│   └── decoder_checkpoints/ # Saved model checkpoints
└── TestNew/               # Testing and evaluation code
    ├── EED.py            # Early Exit Decoding implementation
    ├── SSD.py            # Self-Speculative Decoding implementation
    └── tokenizer.py      # Custom tokenizer
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

## Testing Early Exit and Speculative Decoding

The TestNew directory contains implementations of Early Exit Decoding (EED) and Self-Speculative Decoding (SSD):

1. Early Exit Decoding:
```bash
python TestNew/EED.py
```

2. Self-Speculative Decoding:
```bash
python TestNew/SSD.py
```

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