import pandas as pd
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from main import collate_fn
from dataset import RequestsDataset
from dumb_tokenizer import CharTokenizer
from pln_model import PLNModel
from sklearn.metrics import precision_score, recall_score, f1_score
from tqdm import tqdm

def evaluate_pln(model, dataloader, device, tokenizer, max_samples):
    model.eval()
    correct = 0
    total = 0

    all_preds = []
    all_labels = []

    correct_samples = []
    incorrect_samples = []

    batch_bar = tqdm(enumerate(dataloader), total=len(dataloader), unit="batch", leave=False)
    with torch.no_grad():
        for batch_idx, (inputs, _, _, labels) in batch_bar:
            inputs = inputs.to(device)
            labels = labels.to(device).long()
            
            batch_bar.set_description(f"Batch {batch_idx}")


            cls_logits = model(inputs)  # (B, L*A, 2)
            probs = F.softmax(cls_logits, dim=-1)
            suspicious_probs = probs[:, :, 1].max(dim=1).values
            preds = (suspicious_probs >= 0.5).long()

            all_preds.extend(preds.cpu().tolist())
            all_labels.extend(labels.cpu().tolist())

            correct_mask = (preds == labels)
            correct += correct_mask.sum().item()
            total += labels.size(0)

            for i in range(len(labels)):
                if len(correct_samples) >= max_samples and len(incorrect_samples) >= max_samples:
                    break
                decoded_input = tokenizer.decode(inputs[i].cpu())
                true_label = labels[i].item()
                pred_label = preds[i].item()

                sample = {
                    'input': decoded_input,
                    'true_label': true_label,
                    'pred_label': pred_label,
                }

                if pred_label == true_label and len(correct_samples) < max_samples:
                    correct_samples.append(sample)
                elif pred_label != true_label and len(incorrect_samples) < max_samples:
                    incorrect_samples.append(sample)

            if len(correct_samples) >= max_samples and len(incorrect_samples) >= max_samples:
                break

    accuracy = correct / total if total > 0 else 0
    precision = precision_score(all_labels, all_preds, zero_division=0)
    recall = recall_score(all_labels, all_preds, zero_division=0)
    f1 = f1_score(all_labels, all_preds, zero_division=0)


    return accuracy, precision, recall, f1, correct_samples, incorrect_samples

if __name__ == "__main__":

    # Load dataset
    csv_path = "CISC2010_cleaned_test.csv"
    df = pd.read_csv(csv_path)

    df_tok = pd.read_csv("CISC2010_cleaned_train.csv")

    # Build tokenizer vocab from all content
    tokenizer = CharTokenizer("".join(df_tok["content"].tolist()))

    # Dataset + DataLoader
    dataset = RequestsDataset(df, tokenizer, max_len=1024)
    dataloader = DataLoader(dataset, batch_size=32, shuffle=False, collate_fn=collate_fn)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"running on {device}")

    # Init model
    vocab_size = tokenizer.vocab_size
    embedding_dim = 64
    max_length = 1024
    anchor_sizes = [4, 8, 16]

    model = PLNModel(vocab_size, embedding_dim, max_length, anchor_sizes)

    checkpoint = torch.load('checkpoints/pln_epoch_800.pt', map_location=torch.device('cpu'))

    model.load_state_dict(checkpoint['model_state_dict'])

    model.to(device)

    accuracy, precision, recall, f1, correct_samples, incorrect_samples = evaluate_pln(model, dataloader, device, tokenizer, 15)
    print("\n====== Final Evaluation Metrics ======")
    print(f"Accuracy   : {accuracy:.4f}")
    print(f"Precision  : {precision:.4f}")
    print(f"Recall     : {recall:.4f}")
    print(f"F1 Score   : {f1:.4f}")
    print("======================================\n")

    print("Sample Correct Predictions:")
    for sample in correct_samples:
        print(f"Input      : {sample['input']}")
        print(f"True Label : {sample['true_label']}, Predicted: {sample['pred_label']}")
        print("-----")

    print("\nSample Incorrect Predictions:")
    for sample in incorrect_samples:
        print(f"Input      : {sample['input']}")
        print(f"True Label : {sample['true_label']}, Predicted: {sample['pred_label']}")
        print("-----")
