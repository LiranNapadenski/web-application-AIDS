import torch
from torch.utils.data import DataLoader

def train_pln(model, dataset, optimizer, device, batch_size=32, epochs=10):
    model.to(device)
    model.train()

    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, collate_fn=dataset.collate_fn)

    for epoch in range(epochs):
        total_loss = 0
        for batch in dataloader:
            # unpack batch data
            inputs, masks, gt_boxes, labels = batch

            inputs = inputs.to(device)        # (B, L)
            masks = masks.to(device)          # (B, L)
            # gt_boxes: list of tensors with shape (num_spans, 2) per sample
            labels = labels.to(device)

            optimizer.zero_grad()

            # Forward pass: get predicted anchor classifications and regressions
            cls_scores, reg_preds = model(inputs, masks) 
            # cls_scores: (B, num_anchors, 2)
            # reg_preds: (B, num_anchors, 2)

            # Prepare ground truth for anchors (assign pos/neg/ignore)
            anchor_labels, anchor_regs = assign_anchors_to_gt(cls_scores, reg_preds, gt_boxes)

            # Compute loss
            loss = compute_pln_loss(cls_scores, reg_preds, anchor_labels, anchor_regs)

            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        avg_loss = total_loss / len(dataloader)
        print(f"Epoch {epoch+1}/{epochs} - Loss: {avg_loss:.4f}")
