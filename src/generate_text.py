import torch
import argparse
import logging
from transformer_model import TransformerModel, load_wikitext_data, build_vocab
from datasets import load_dataset
import os

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
    """Load model and configuration from checkpoint."""
    logging.info(f"Loading checkpoint from {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location=device)
    
    # Get vocabulary size from dataset
    dataset = load_dataset("wikitext", "wikitext-2-raw-v1")
    vocab = build_vocab(dataset)
    vocab_size = len(vocab)
    
    config = checkpoint['config']
    model = TransformerModel(
        vocab_size=vocab_size,  # Use vocab size from dataset
        d_model=config['d_model'],
        num_layers=config['num_layers'],
        num_heads=config['num_heads'],
        d_ff=config['d_ff'],
        max_seq_length=config['seq_length'],
        dropout=config['dropout'],
        max_skip_prob=0.0  # No layer skipping during inference
    ).to(device)
    
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    
    logging.info("Model configuration:")
    for k, v in config.items():
        logging.info(f"{k}: {v}")
    
    return model, vocab

def generate_text(model, input_text, vocab, max_length=50, temperature=1.0, device='cpu'):
    """Generate text continuation from input prompt."""
    model.eval()
    
    # Tokenize input text
    tokens = input_text.split()
    input_ids = [vocab.get(token, vocab['<unk>']) for token in tokens]
    input_tensor = torch.tensor([input_ids], dtype=torch.long).to(device)
    
    generated = input_ids.copy()
    
    with torch.no_grad():
        for _ in range(max_length):
            # Take last sequence_length tokens as input
            curr_input = torch.tensor([generated[-100:]], dtype=torch.long).to(device)
            
            # Get model predictions
            output = model(curr_input)
            next_token_logits = output[0, -1, :] / temperature
            
            # Sample from the distribution
            probs = torch.softmax(next_token_logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1).item()
            
            generated.append(next_token)
            
            # Stop if we generate end token (if defined)
            if next_token == vocab.get('<eos>', None):
                break
    
    # Convert tokens back to text
    id_to_token = {v: k for k, v in vocab.items()}
    generated_text = ' '.join(id_to_token[token] for token in generated)
    
    return generated_text

def main():
    parser = argparse.ArgumentParser(description="Generate text using trained Transformer model")
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to model checkpoint")
    parser.add_argument("--prompt", type=str, default="The story begins", help="Text prompt to start generation")
    parser.add_argument("--max_length", type=int, default=50, help="Maximum length of generated text")
    parser.add_argument("--temperature", type=float, default=1.0, help="Sampling temperature (higher = more random)")
    args = parser.parse_args()

    setup_logging()
    
    # Setup device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logging.info(f"Using device: {device}")
    
    # Load model and vocabulary
    model, vocab = load_model_from_checkpoint(args.checkpoint, device)
    
    # Generate text
    logging.info(f"\nPrompt: {args.prompt}")
    generated_text = generate_text(
        model, 
        args.prompt, 
        vocab, 
        max_length=args.max_length,
        temperature=args.temperature,
        device=device
    )
    
    logging.info("\nGenerated text:")
    logging.info("=" * 50)
    logging.info(generated_text)
    logging.info("=" * 50)

if __name__ == "__main__":
    main() 