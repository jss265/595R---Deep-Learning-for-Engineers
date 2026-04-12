import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split

# Import custom classes
from WorkUp_DATA import IMURepDataset, pad_collate
from WorkUp_GRU import RNNPredictor

def train_and_eval(lists):
    # 1. Setup Device (MPS for Mac, CUDA for Nvidia, otherwise CPU)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # 2. Load the Dataset
    data_dir = os.path.join(os.path.dirname(__file__), "ESP32-C3 IMU", "Python Files", "recordings", "training1")
    print(f"Loading dataset from: {data_dir}")
    full_dataset = IMURepDataset(data_dir, *lists)
    print(f"Total labeled sequences loaded: {len(full_dataset)}")

    # 3. Split the Data (80% Training, 20% Testing)
    train_size = int(0.8 * len(full_dataset))
    test_size = len(full_dataset) - train_size
    train_dataset, test_dataset = random_split(full_dataset, [train_size, test_size])

    # 4. Create DataLoaders (using our custom collate_fn for the padding)
    batch_size = 32
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, collate_fn=pad_collate)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, collate_fn=pad_collate)

    # 5. Initialize the Model, Loss Function, and Optimizer
    model = RNNPredictor(input_size=10, hidden_size=32, num_classes=4).to(device)
    
    # CrossEntropyLoss expects discrete integer targets and raw unnormalized logits (this is AI jargon, but it works hehe)
    loss_fn = nn.CrossEntropyLoss() # aka criterion
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    epochs = 40
    print("\nStarting Training...\n" + "-"*50)

    # 6. The Training Loop
    for epoch in range(epochs):
        model.train()
        total_train_loss = 0
        correct_train = 0
        
        for sequences, labels, lengths in train_loader:
            sequences, labels = sequences.to(device), labels.to(device)

            # Reset gradients
            optimizer.zero_grad()
            
            # Forward pass (lengths stays on CPU as enforced inside the model)
            outputs = model(sequences, lengths)
            
            # Calculate loss
            loss = loss_fn(outputs, labels)
            
            # Backward pass & Optimize
            loss.backward()
            optimizer.step()

            # Track metrics
            total_train_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1) # this is the classification prediction (0, 1, 2, or 3)
            correct_train += (predicted == labels).sum().item()

        # 7. Evaluation Phase (No backpropagation)
        model.eval()
        total_test_loss = 0
        correct_test = 0
        
        with torch.no_grad():
            for sequences, labels, lengths in test_loader:
                sequences, labels = sequences.to(device), labels.to(device)
                
                outputs = model(sequences, lengths)
                loss = loss_fn(outputs, labels)
                total_test_loss += loss.item()
                
                _, predicted = torch.max(outputs.data, 1) # this is the classification prediction (0, 1, 2, or 3)
                correct_test += (predicted == labels).sum().item()

        # Print statistics
        train_acc = 100 * correct_train / train_size
        test_acc = 100 * correct_test / test_size
        avg_train_loss = total_train_loss / len(train_loader)
        avg_test_loss = total_test_loss / len(test_loader)

        if (epoch + 1) % 5 == 0 or epoch == 0:
            print(f"Epoch [{epoch+1:2d}/{epochs}] "
                  f"| Train Loss: {avg_train_loss:.4f} | Train Acc: {train_acc:5.2f}% "
                  f"| Test Loss: {avg_test_loss:.4f} | Test Acc: {test_acc:5.2f}%")

if __name__ == "__main__":
    train_and_eval()