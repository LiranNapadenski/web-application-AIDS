import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm
import pandas as pd

from PLNTrainer import train_pln
from dumb_tokenizer import CharTokenizer          # Your CharTokenizer class
from dataset import RequestsDataset          # Dataset using CharTokenizer
from pln_model import PLNModel                    # PLNModel code
from lossFunction import request_loss                 # Your request-level loss function

def collate_fn(batch):
    seqs, masks, gt_boxes, labels = zip(*batch)
    lengths = torch.tensor([s.size(0) for s in seqs], dtype=torch.long)
    max_len = lengths.max().item()
    padded_seqs = torch.full((len(seqs), max_len), fill_value=0, dtype=torch.long)
    padded_masks = torch.zeros_like(padded_seqs, dtype=torch.bool)
    for i, s in enumerate(seqs):
        padded_seqs[i, : s.size(0)] = s
        padded_masks[i, : s.size(0)] = masks[i]
    labels = torch.tensor(labels, dtype=torch.long)
    return padded_seqs, padded_masks, None, labels

def main():
    # Load dataset
    csv_path = "CISC2010_cleaned_train.csv"
    df = pd.read_csv(csv_path)

    # Build tokenizer vocab from all content
    tokenizer = CharTokenizer("".join(df["content"].tolist()))

    # Dataset + DataLoader
    dataset = RequestsDataset(df, tokenizer, max_len=1024)
    dataloader = DataLoader(dataset, batch_size=16, shuffle=True, collate_fn=collate_fn)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"running on {device}")

    # Init model
    vocab_size = tokenizer.vocab_size
    embedding_dim = 64
    max_length = 1024
    anchor_sizes = [4, 8, 16]

    model = PLNModel(vocab_size, embedding_dim, max_length, anchor_sizes)

    # Optimizer
    optimizer = optim.Adam(model.parameters(), lr=1e-4)

    # Train
    train_pln(model, dataloader, optimizer, device, epochs=800, save_every=100)

if __name__ == "__main__":
    main()