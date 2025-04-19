import torch
import argparse
import logging
import os
from decoder_model import DecoderOnlyTransformer, load_wikitext_data, build_vocab
from datasets import load_dataset

def setup_logging():
    os.makedirs('logs', exist_ok=True)
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s | %(message)s',
        handlers=[
            logging.StreamHandler()
        ]
    )

def load_model_from_checkpoint(checkpoint_path, device):
    logging.info(f"Loading checkpoint from {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location=device)

    dataset = load_dataset("wikitext", "wikitext-2-raw-v1")
    vocab = build_vocab(dataset)
    vocab_size = len(vocab)

    model = DecoderOnlyTransformer(
        vocab_size=vocab_size,
        d_model=checkpoint['model_state_dict']['embedding.weight'].size(1),
        num_layers=len([k for k in checkpoint['model_state_dict'] if k.startswith('layers.') and '.attn.qkv_proj.weight' in k]),
        num_heads=8,  # Use consistent default if not saved
        d_ff=512,     # Adjust based on training config
        max_seq_len=100,
        dropout=0.0,
        max_skip_prob=0.0
    ).to(device)

    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()

    return model, vocab

def generate_text(model, input_text, vocab, max_length=50, temperature=1.0, device='cpu'):
    model.eval()
    tokens = input_text.lower().split()
    input_ids = [vocab.get(token, vocab['<unk>']) for token in tokens]
    generated = input_ids.copy()

    with torch.no_grad():
        for _ in range(max_length):
            curr_input = torch.tensor([generated[-100:]], dtype=torch.long).to(device)
            output = model(curr_input)
            next_token_logits = output[0, -1, :] / temperature
            probs = torch.softmax(next_token_logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1).item()
            generated.append(next_token)
            if next_token == vocab.get('<eos>', None):
                break

    id_to_token = {v: k for k, v in vocab.items()}
    generated_text = ' '.join(id_to_token.get(token, '<unk>') for token in generated)
    return generated_text

def main():
    parser = argparse.ArgumentParser(description="Generate text using Decoder-Only Transformer")
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to model checkpoint")
    parser.add_argument("--prompt", type=str, default="The Sinclair Scientific Programmable was introduced in 1975 , with the same case as the Sinclair Oxford.", help="Initial text prompt")
    parser.add_argument("--max_length", type=int, default=100, help="Max length of generated text")
    parser.add_argument("--temperature", type=float, default=1.0, help="Sampling temperature")
    args = parser.parse_args()

    setup_logging()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logging.info(f"Using device: {device}")

    model, vocab = load_model_from_checkpoint(args.checkpoint, device)
    logging.info(f"Prompt: {args.prompt}")
    generated_text = generate_text(model, args.prompt, vocab, args.max_length, args.temperature, device)

    logging.info("\nGenerated text:")
    logging.info("=" * 50)
    logging.info(generated_text)
    logging.info("=" * 50)

if __name__ == "__main__":
    main()
