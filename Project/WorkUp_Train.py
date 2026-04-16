import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Subset

from WorkUp_DATA import IMURepDataset, pad_collate
from WorkUp_GRU import RNNPredictor

def stratified_split(dataset, train_fraction=0.8):
    label_to_indices = {}

    for index, label in enumerate(dataset.labels):
        label_to_indices.setdefault(int(label.item()), []).append(index)

    train_indices = []
    test_indices = []

    for label in sorted(label_to_indices):
        indices = label_to_indices[label]
        shuffled_positions = torch.randperm(len(indices)).tolist()
        shuffled_indices = [indices[position] for position in shuffled_positions]
        split_point = int(train_fraction * len(shuffled_indices))
        train_indices.extend(shuffled_indices[:split_point])
        test_indices.extend(shuffled_indices[split_point:])

    return Subset(dataset, train_indices), Subset(dataset, test_indices)

def filter_subset_by_labels(dataset, allowed_labels):
    allowed_labels = set(allowed_labels)

    if isinstance(dataset, Subset):
        base_dataset = dataset.dataset
        indices = [idx for idx in dataset.indices if int(base_dataset.labels[idx].item()) in allowed_labels]
        return Subset(base_dataset, indices)

    indices = [idx for idx, label in enumerate(dataset.labels) if int(label.item()) in allowed_labels]
    return Subset(dataset, indices)

def make_loader(dataset, batch_size, shuffle):
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, collate_fn=pad_collate)

def evaluate(model, loader, loss_fn, device):
    model.eval()
    total_loss = 0.0
    correct = 0
    total_samples = len(loader.dataset)

    with torch.no_grad():
        for sequences, labels, lengths in loader:
            sequences, labels = sequences.to(device), labels.to(device)
            outputs = model(sequences, lengths)
            loss = loss_fn(outputs, labels)
            total_loss += loss.item()
            predicted = outputs.argmax(dim=1)
            correct += (predicted == labels).sum().item()

    avg_loss = total_loss / max(len(loader), 1)
    accuracy = 100.0 * correct / max(total_samples, 1)
    return avg_loss, accuracy

def train_stage(model, train_loader, test_loader, device, epochs, lr, stage_name):
    loss_fn = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)

    print(f"\n{stage_name}\n{'-' * len(stage_name)}")

    for epoch in range(epochs):
        model.train()
        total_train_loss = 0.0
        correct_train = 0
        total_train_samples = len(train_loader.dataset)

        for sequences, labels, lengths in train_loader:
            sequences, labels = sequences.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(sequences, lengths)
            loss = loss_fn(outputs, labels)
            loss.backward()
            optimizer.step()

            total_train_loss += loss.item()
            predicted = outputs.argmax(dim=1)
            correct_train += (predicted == labels).sum().item()

        avg_train_loss = total_train_loss / max(len(train_loader), 1)
        train_acc = 100.0 * correct_train / max(total_train_samples, 1)
        avg_test_loss, test_acc = evaluate(model, test_loader, loss_fn, device)

        if epoch == 0 or (epoch + 1) % 25 == 0 or epoch + 1 == epochs:
            print(
                f"Epoch [{epoch + 1:3d}/{epochs}] "
                f"| Train Loss: {avg_train_loss:.4f} | Train Acc: {train_acc:5.2f}% "
                f"| Test Loss: {avg_test_loss:.4f} | Test Acc: {test_acc:5.2f}%"
            )

def train_and_eval():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    data_dir = os.path.join(os.path.dirname(__file__), "ESP32-C3 IMU", "Python Files", "recordings", "training1")
    print(f"Loading dataset from: {data_dir}")
    full_dataset = IMURepDataset(data_dir)
    print(f"Total labeled sequences loaded: {len(full_dataset)}")

    train_dataset, test_dataset = stratified_split(full_dataset)
    print(f"Train split size: {len(train_dataset)}")
    print(f"Test split size: {len(test_dataset)}")

    batch_size = 32
    model = RNNPredictor(input_size=10, hidden_size=32, num_classes=4).to(device)

    stages = [
        {"name": "Stage 1: reps 0, 1", "allowed_labels": [0, 1], "epochs": 150, "lr": 1e-3},
        {"name": "Stage 2: reps 0, 1, 2", "allowed_labels": [0, 1, 2], "epochs": 325, "lr": 5e-4},
        {"name": "Stage 3: reps 0, 1, 2, 3", "allowed_labels": [0, 1, 2, 3], "epochs": 750, "lr": 1e-4},
    ]

    print("\nStarting Curriculum Training...\n" + "-" * 50)

    for stage in stages:
        stage_train_dataset = filter_subset_by_labels(train_dataset, stage["allowed_labels"])
        stage_test_dataset = filter_subset_by_labels(test_dataset, stage["allowed_labels"])
        stage_train_loader = make_loader(stage_train_dataset, batch_size=batch_size, shuffle=True)
        stage_test_loader = make_loader(stage_test_dataset, batch_size=batch_size, shuffle=False)

        print(f"{stage['name']} | Train samples: {len(stage_train_dataset)} | Test samples: {len(stage_test_dataset)}")
        train_stage(
            model=model,
            train_loader=stage_train_loader,
            test_loader=stage_test_loader,
            device=device,
            epochs=stage["epochs"],
            lr=stage["lr"],
            stage_name=stage["name"],
        )
    print('\n Done Training.')

if __name__ == "__main__":
    train_and_eval()