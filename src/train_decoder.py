import torch
import torch.nn as nn
import argparse
import logging
import os
import time
from datetime import datetime
from decoder_model import DecoderOnlyTransformer, load_wikitext_data

def setup_logging(args):
    os.makedirs('decoder_logs', exist_ok=True)
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    mode = '_'.join(filter(None, ['train' if args.train else '', 'eval' if args.eval else '']))
    log_file = f'decoder_logs/decoder_{mode}_{timestamp}.log'
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s | %(message)s',
        handlers=[logging.FileHandler(log_file), logging.StreamHandler()]
    )
    logging.info(f"Logging to: {log_file}")
    return log_file

def train_model(model, train_loader, val_loader, vocab_size, device, epochs=1, lr=1e-4):
    logging.info("=== Starting Training ===")
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    os.makedirs('decoder_checkpoints', exist_ok=True)
    best_val_loss = float('inf')
    best_model_state = None
    best_epoch = 0
    start_time = time.time()

    for epoch in range(epochs):
        model.train()
        epoch_loss = 0.0
        for batch_idx, (x, y) in enumerate(train_loader, 1):
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            output = model(x)
            loss = criterion(output.view(-1, vocab_size), y.view(-1))
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()

            if batch_idx % 100 == 0:
                logging.info(f"Epoch [{epoch+1}/{epochs}] Batch [{batch_idx}] Loss: {loss.item():.4f}")

        avg_train_loss = epoch_loss / len(train_loader)
        val_loss = evaluate_model(model, val_loader, vocab_size, device)

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_model_state = model.state_dict().copy()
            best_epoch = epoch + 1
            logging.info(f"New best model with val loss: {val_loss:.4f}")

        logging.info(f"Epoch {epoch+1} | Train Loss: {avg_train_loss:.4f} | Val Loss: {val_loss:.4f}")

    if best_model_state:
        checkpoint_path = (
            f"decoder_checkpoints/best_decoder_d{model.embedding.embedding_dim}_h{model.layers[0].attn.num_heads}_"
            f"l{len(model.layers)}_ff{model.layers[0].ff.w1.out_features}_dp{model.layers[0].dropout1.p}_"
            f"ls{int(model.layers[-1].skip_prob * 100)}_e{best_epoch}_val{best_val_loss:.4f}.pt"
        )
        torch.save({
            'model_state_dict': best_model_state,
            'val_loss': best_val_loss,
            'epoch': best_epoch
        }, checkpoint_path)
        logging.info(f"Saved best model to {checkpoint_path}")

    logging.info(f"Training completed in {time.time() - start_time:.2f}s")
    return best_val_loss

def evaluate_model(model, val_loader, vocab_size, device):
    model.eval()
    total_loss = 0.0
    criterion = nn.CrossEntropyLoss()
    with torch.no_grad():
        for x, y in val_loader:
            x, y = x.to(device), y.to(device)
            output = model(x)
            loss = criterion(output.view(-1, vocab_size), y.view(-1))
            total_loss += loss.item()
    return total_loss / len(val_loader)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--train", action="store_true")
    parser.add_argument("--eval", action="store_true")
    parser.add_argument("--checkpoint", type=str)
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--seq_length", type=int, default=100)
    parser.add_argument("--d_model", type=int, default=128)
    parser.add_argument("--num_layers", type=int, default=6)
    parser.add_argument("--num_heads", type=int, default=8)
    parser.add_argument("--d_ff", type=int, default=512)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--max_skip_prob", type=float, default=0.0)
    args = parser.parse_args()

    setup_logging(args)
    train_loader, val_loader, vocab_size = load_wikitext_data(args.seq_length, args.batch_size)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = DecoderOnlyTransformer(
        vocab_size=vocab_size,
        d_model=args.d_model,
        num_layers=args.num_layers,
        num_heads=args.num_heads,
        d_ff=args.d_ff,
        max_seq_len=args.seq_length,
        dropout=args.dropout,
        max_skip_prob=args.max_skip_prob
    ).to(device)

    if args.checkpoint:
        checkpoint = torch.load(args.checkpoint)
        model.load_state_dict(checkpoint['model_state_dict'])
        logging.info(f"Loaded checkpoint from {args.checkpoint} at epoch {checkpoint['epoch']}")

    if args.train:
        train_model(model, train_loader, val_loader, vocab_size, device, args.epochs, args.lr)

    if args.eval:
        val_loss = evaluate_model(model, val_loader, vocab_size, device)
        logging.info(f"Validation Loss: {val_loss:.4f}")

    logging.info("Done!")

if __name__ == "__main__":
    main()
