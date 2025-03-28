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
    best_model_state = None
    best_optimizer_state = None
    best_epoch = 0
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
        
        # Keep track of best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_model_state = model.state_dict().copy()
            best_optimizer_state = optimizer.state_dict().copy()
            best_epoch = epoch + 1
            logging.info(f"New best model found with validation loss: {val_loss:.4f}")
        
        epoch_time = time.time() - epoch_start_time
        logging.info(
            f"Epoch {epoch+1}/{epochs} completed | "
            f"Train Loss: {avg_train_loss:.4f} | "
            f"Val Loss: {val_loss:.4f} | "
            f"Time: {epoch_time:.2f}s"
        )
    
    # Save only the best model at the end of training
    if best_model_state is not None:
        checkpoint = {
            'epoch': best_epoch,
            'model_state_dict': best_model_state,
            'optimizer_state_dict': best_optimizer_state,
            'val_loss': best_val_loss,
            'config': {
                'd_model': model.encoder.layers[0].self_attn.d_model,
                'num_heads': model.encoder.layers[0].self_attn.num_heads,
                'd_ff': model.encoder.layers[0].feed_forward.linear1.out_features,
                'num_layers': len(model.encoder.layers),
                'dropout': model.dropout.p,
                'max_skip_prob': model.max_skip_prob,
                'seq_length': model.max_seq_length
            }
        }
        checkpoint_path = (
            f'checkpoints/best_model_'
            f'd{model.encoder.layers[0].self_attn.d_model}_'
            f'h{model.encoder.layers[0].self_attn.num_heads}_'
            f'l{len(model.encoder.layers)}_'
            f'ff{model.encoder.layers[0].feed_forward.linear1.out_features}_'
            f'dp{model.dropout.p}_'
            f'ls{int(model.max_skip_prob*100)}_'
            f'e{best_epoch}_'
            f'val{best_val_loss:.4f}.pt'
        )
        torch.save(checkpoint, checkpoint_path)
        logging.info(f"Saved best model to {checkpoint_path}")
    
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
    
    # New model architecture arguments
    parser.add_argument("--seq_length", type=int, default=100, help="Sequence length for training")
    parser.add_argument("--d_model", type=int, default=128, help="Model dimension")
    parser.add_argument("--num_layers", type=int, default=6, help="Number of transformer layers")
    parser.add_argument("--num_heads", type=int, default=8, help="Number of attention heads")
    parser.add_argument("--d_ff", type=int, default=512, help="Feed-forward network dimension")
    parser.add_argument("--dropout", type=float, default=0.1, help="Dropout rate")
    parser.add_argument("--max_skip_prob", type=float, default=0.5, 
                       help="Maximum layer skip probability (only used when layerskip is enabled)")
    
    args = parser.parse_args()

    # Setup logging
    log_file = setup_logging(args)

    # Use args instead of hardcoded values
    seq_length = args.seq_length
    d_model = args.d_model
    num_layers = args.num_layers
    num_heads = args.num_heads
    d_ff = args.d_ff
    dropout = args.dropout
    max_skip_prob = args.max_skip_prob if args.layerskip else 0.0

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
