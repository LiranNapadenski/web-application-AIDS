import os
import torch
from torch.utils.data import DataLoader
from tqdm.auto import tqdm
from lossFunction import request_loss


def train_pln(model,
              dataloader,
              optimizer,
              device,
              batch_size: int = 32,
              epochs: int = 10,
              save_every: int = 5000,
              save_dir: str = "checkpoints"):

    os.makedirs(save_dir, exist_ok=True)
    model.to(device)
    epoch_bar = tqdm(range(1, epochs + 1), desc="Epochs", unit="epoch")

    for epoch in epoch_bar:
        model.train()
        running_loss = 0.0

        batch_bar = tqdm(dataloader,
                         desc=f"Epoch {epoch}",
                         unit="batch",
                         leave=False)

        for inputs, _, _, labels in batch_bar:
            inputs = inputs.to(device)           # (B, L)
            labels = labels.to(device).long()    # (B,)

            optimizer.zero_grad()
            cls_logits = model(inputs)           # (B, L*A, 2)
            loss = request_loss(cls_logits, labels)

            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            batch_bar.set_postfix(loss=f"{loss.item():.4f}")

        avg_loss = running_loss / len(dataloader)
        epoch_bar.set_postfix(avg_loss=f"{avg_loss:.4f}")

        # ── checkpoint every `save_every` epochs ───────────────────────────
        if epoch % save_every == 0:
            ckpt_path = os.path.join(save_dir, f"pln_epoch_{epoch}.pt")
            torch.save(
                {
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                },
                ckpt_path,
            )
            tqdm.write(f"✓ Saved checkpoint → {ckpt_path}")

