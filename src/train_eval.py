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

def train_model(model, train_loader, val_loader, vocab_size, device, epochs=1, lr=1e-4):
    logging.info("=== Starting Training ===")
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    
    # Create checkpoints directory
    os.makedirs('checkpoints', exist_ok=True)
    
    best_val_loss = float('inf')
    total_steps = len(train_loader)
    start_time = time.time()
    
    for epoch in range(epochs):
        # Training phase
        model.train()
        epoch_loss = 0.0
        epoch_start_time = time.time()
        
        for batch_idx, (x, y) in enumerate(train_loader, 1):
            batch_start = time.time()
            
            x = x.to(device)
            y = y.to(device)
            
            optimizer.zero_grad()
            output = model(x)
            loss = criterion(output.view(-1, vocab_size), y.view(-1))
            loss.backward()
            optimizer.step()
            
            batch_loss = loss.item()
            epoch_loss += batch_loss
            
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
        
        avg_train_loss = epoch_loss / total_steps
        
        # Validation phase
        val_loss = evaluate_model(model, val_loader, vocab_size, device)
        
        # Save checkpoint if validation loss improved
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            checkpoint = {
                'epoch': epoch + 1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'train_loss': avg_train_loss,
                'val_loss': val_loss,
                'config': {
                    'd_model': model.encoder.layers[0].self_attn.d_model,
                    'num_heads': model.encoder.layers[0].self_attn.num_heads,
                    'd_ff': model.encoder.layers[0].feed_forward.linear1.out_features,
                    'num_layers': len(model.encoder.layers),
                    'dropout': model.dropout.p,
                }
            }
            checkpoint_path = f'checkpoints/model_epoch{epoch+1}_valloss{val_loss:.4f}.pt'
            torch.save(checkpoint, checkpoint_path)
            logging.info(f"Saved checkpoint to {checkpoint_path}")
        
        epoch_time = time.time() - epoch_start_time
        logging.info(
            f"Epoch {epoch+1}/{epochs} completed | "
            f"Train Loss: {avg_train_loss:.4f} | "
            f"Val Loss: {val_loss:.4f} | "
            f"Time: {epoch_time:.2f}s"
        )
    
    total_time = time.time() - start_time
    logging.info(f"Training completed in {total_time:.2f}s")
    return best_val_loss

def evaluate_model(model, val_loader, vocab_size, device):
    model.eval()
    criterion = nn.CrossEntropyLoss()
    total_loss = 0.0
    total_steps = len(val_loader)
    
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
    return final_avg_loss

def load_checkpoint(model, optimizer, checkpoint_path):
    checkpoint = torch.load(checkpoint_path)
    model.load_state_dict(checkpoint['model_state_dict'])
    if optimizer is not None:
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    return checkpoint['epoch'], checkpoint['train_loss'], checkpoint['val_loss']

def main():
    parser = argparse.ArgumentParser(description="Train/Evaluate Transformer model on WikiText-2")
    parser.add_argument("--layerskip", action="store_true", help="Enable layer dropout")
    parser.add_argument("--train", action="store_true", help="Run training")
    parser.add_argument("--eval", action="store_true", help="Run evaluation")
    parser.add_argument("--epochs", type=int, default=1, help="Number of epochs")
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size")
    parser.add_argument("--lr", type=float, default=1e-4, help="Learning rate")
    parser.add_argument("--checkpoint", type=str, help="Path to checkpoint to resume from")
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

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    # Load checkpoint if specified
    start_epoch = 0
    if args.checkpoint:
        logging.info(f"Loading checkpoint from {args.checkpoint}")
        start_epoch, train_loss, val_loss = load_checkpoint(model, optimizer, args.checkpoint)
        logging.info(f"Resumed from epoch {start_epoch} with train_loss: {train_loss:.4f}, val_loss: {val_loss:.4f}")

    if args.train:
        train_model(model, train_loader, val_loader, vocab_size, device, args.epochs, args.lr)

    if args.eval:
        val_loss = evaluate_model(model, val_loader, vocab_size, device)
        logging.info(f"Final validation loss: {val_loss:.4f}")

    logging.info("Done!")

if __name__ == "__main__":
    main()
