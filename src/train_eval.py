import torch
import torch.nn as nn
import argparse
from transformer_model import TransformerModel, load_wikitext_data

def train_model(model, train_loader, vocab_size, device, epochs=1, lr=1e-4):
    print("Starting training...")
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    for epoch in range(epochs):
        model.train()
        epoch_loss = 0.0
        for x, y in train_loader:
            # Move data to device
            x = x.to(device)
            y = y.to(device)
            
            optimizer.zero_grad()
            output = model(x)  # shape: (batch_size, seq_length, vocab_size)
            loss = criterion(output.view(-1, vocab_size), y.view(-1))
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
        avg_loss = epoch_loss / len(train_loader)
        print(f"Epoch {epoch+1}/{epochs}, Loss: {avg_loss:.4f}")

def evaluate_model(model, val_loader, vocab_size, device):
    print("Starting evaluation...")
    model.eval()
    total_loss = 0.0
    criterion = nn.CrossEntropyLoss()
    with torch.no_grad():
        for x, y in val_loader:
            # Move data to device
            x = x.to(device)
            y = y.to(device)
            
            output = model(x)
            loss = criterion(output.view(-1, vocab_size), y.view(-1))
            total_loss += loss.item()
    avg_loss = total_loss / len(val_loader)
    print("Evaluation Loss:", avg_loss)

def main():
    parser = argparse.ArgumentParser(description="Train/Evaluate a Transformer model on WikiText-2 with optional layer dropout.")
    parser.add_argument("--layerskip", action="store_true", help="Enable layer dropout (stochastic depth).")
    parser.add_argument("--train", action="store_true", help="Run training.")
    parser.add_argument("--eval", action="store_true", help="Run evaluation.")
    args = parser.parse_args()

    # Data parameters
    seq_length = 100
    batch_size = 32

    # Load WikiText-2 data using HuggingFace's datasets
    train_loader, val_loader, vocab_size = load_wikitext_data(seq_length, batch_size)

    # Model hyperparameters
    d_model = 128
    num_layers = 6
    num_heads = 8
    d_ff = 512
    dropout = 0.1
    max_skip_prob = 0.5 if args.layerskip else 0.0

    print("Layer dropout enabled." if args.layerskip else "Layer dropout disabled.")
    print(f"Vocabulary size: {vocab_size}")

    # Device: use GPU if available
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Create the Transformer model and move it to the device
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

    # If no mode is specified, run a single forward pass for demonstration.
    if not args.train and not args.eval:
        print("No mode specified. Running a single forward pass for demonstration.")
        for x, _ in train_loader:
            x = x.to(device)
            output = model(x)
            print("Output shape:", output.shape)
            break
        return

    if args.train:
        train_model(model, train_loader, vocab_size, device, epochs=1, lr=1e-4)

    if args.eval:
        evaluate_model(model, val_loader, vocab_size, device)

if __name__ == "__main__":
    main()
