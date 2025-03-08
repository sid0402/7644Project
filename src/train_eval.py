import torch
import torch.nn as nn
import argparse
import logging
import os
import time
from datetime import datetime
from transformer_model import TransformerModel, load_wikitext_data

def setup_logging(args):
    # Create logs directory if it doesn't exist
    os.makedirs('logs', exist_ok=True)
    
    # Create timestamp for unique log file
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    mode = '_'.join(mode for mode in ['train' if args.train else None, 
                                    'eval' if args.eval else None] 
                    if mode is not None)
    log_file = f'logs/transformer_{mode}_{timestamp}.log'
    
    # Configure logging to write to both file and console
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s | %(message)s',
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler()
        ]
    )
    logging.info(f"Logging to: {log_file}")
    return log_file

def train_model(model, train_loader, vocab_size, device, epochs=1, lr=1e-4):
    logging.info("=== Starting Training ===")
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    
    total_steps = len(train_loader)
    start_time = time.time()
    
    for epoch in range(epochs):
        model.train()
        epoch_loss = 0.0
        epoch_start_time = time.time()
        
        for batch_idx, (x, y) in enumerate(train_loader, 1):
            batch_start = time.time()
            
            # Move data to device
            x = x.to(device)
            y = y.to(device)
            
            optimizer.zero_grad()
            output = model(x)
            loss = criterion(output.view(-1, vocab_size), y.view(-1))
            loss.backward()
            optimizer.step()
            
            batch_loss = loss.item()
            epoch_loss += batch_loss
            
            # Log every 100 batches
            if batch_idx % 100 == 0:
                avg_loss = epoch_loss / batch_idx
                progress = (batch_idx / total_steps) * 100
                batch_time = time.time() - batch_start
                logging.info(
                    f"Epoch: {epoch+1}/{epochs} | "
                    f"Batch: {batch_idx}/{total_steps} ({progress:.1f}%) | "
                    f"Loss: {batch_loss:.4f} | "
                    f"Avg Loss: {avg_loss:.4f} | "
                    f"Batch Time: {batch_time:.2f}s"
                )
        
        # End of epoch logging
        epoch_time = time.time() - epoch_start_time
        avg_epoch_loss = epoch_loss / total_steps
        logging.info(
            f"Epoch {epoch+1}/{epochs} completed | "
            f"Avg Loss: {avg_epoch_loss:.4f} | "
            f"Time: {epoch_time:.2f}s"
        )
    
    total_time = time.time() - start_time
    logging.info(f"Training completed in {total_time:.2f}s")

def evaluate_model(model, val_loader, vocab_size, device):
    logging.info("=== Starting Evaluation ===")
    model.eval()
    criterion = nn.CrossEntropyLoss()
    total_loss = 0.0
    total_steps = len(val_loader)
    start_time = time.time()
    
    with torch.no_grad():
        for batch_idx, (x, y) in enumerate(val_loader, 1):
            x = x.to(device)
            y = y.to(device)
            
            output = model(x)
            loss = criterion(output.view(-1, vocab_size), y.view(-1))
            batch_loss = loss.item()
            total_loss += batch_loss
            
            if batch_idx % 100 == 0:
                progress = (batch_idx / total_steps) * 100
                avg_loss = total_loss / batch_idx
                logging.info(
                    f"Eval Progress: {batch_idx}/{total_steps} ({progress:.1f}%) | "
                    f"Current Loss: {batch_loss:.4f} | "
                    f"Avg Loss: {avg_loss:.4f}"
                )
    
    final_avg_loss = total_loss / total_steps
    total_time = time.time() - start_time
    logging.info(
        f"Evaluation completed | "
        f"Final Loss: {final_avg_loss:.4f} | "
        f"Time: {total_time:.2f}s"
    )

def main():
    parser = argparse.ArgumentParser(description="Train/Evaluate Transformer model on WikiText-2")
    parser.add_argument("--layerskip", action="store_true", help="Enable layer dropout")
    parser.add_argument("--train", action="store_true", help="Run training")
    parser.add_argument("--eval", action="store_true", help="Run evaluation")
    parser.add_argument("--epochs", type=int, default=1, help="Number of epochs")
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size")
    parser.add_argument("--lr", type=float, default=1e-4, help="Learning rate")
    args = parser.parse_args()

    # Setup logging
    log_file = setup_logging(args)

    # Log all hyperparameters
    seq_length = 100
    d_model = 128
    num_layers = 6
    num_heads = 8
    d_ff = 512
    dropout = 0.1
    max_skip_prob = 0.5 if args.layerskip else 0.0

    logging.info("=== Configuration ===")
    logging.info(f"Sequence length: {seq_length}")
    logging.info(f"Batch size: {args.batch_size}")
    logging.info(f"Learning rate: {args.lr}")
    logging.info(f"Epochs: {args.epochs}")
    logging.info(f"Model dimension: {d_model}")
    logging.info(f"Number of layers: {num_layers}")
    logging.info(f"Number of heads: {num_heads}")
    logging.info(f"Feed-forward dim: {d_ff}")
    logging.info(f"Dropout: {dropout}")
    logging.info(f"Layer skip probability: {max_skip_prob}")
    logging.info(f"Layer dropout: {'enabled' if args.layerskip else 'disabled'}")

    # Load data
    logging.info("Loading WikiText-2 dataset...")
    train_loader, val_loader, vocab_size = load_wikitext_data(seq_length, args.batch_size)
    logging.info(f"Vocabulary size: {vocab_size}")

    # Setup device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logging.info(f"Using device: {device}")

    # Create model
    logging.info("Initializing Transformer model...")
    model = TransformerModel(
        vocab_size=vocab_size,
        d_model=d_model,
        num_layers=num_layers,
        num_heads=num_heads,
        d_ff=d_ff,
        max_seq_length=seq_length,
        dropout=dropout,
        max_skip_prob=max_skip_prob
    ).to(device)

    # Log model parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logging.info(f"Total parameters: {total_params:,}")
    logging.info(f"Trainable parameters: {trainable_params:,}")

    if args.train:
        train_model(model, train_loader, vocab_size, device, args.epochs, args.lr)

    if args.eval:
        evaluate_model(model, val_loader, vocab_size, device)

    logging.info("Done!")

if __name__ == "__main__":
    main()
