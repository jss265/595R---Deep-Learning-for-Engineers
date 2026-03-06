# %%
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

# %%

# Reproducibility
torch.manual_seed(42)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f'Using device: {device}')

# %%

datasets = torch.load('bearing_samples2.pt', weights_only=False)  # saved the data to be used later. "torch.save"/"torch.load". Format is .pt (pytorch)
train_ds = datasets['train']
test_ds   = datasets['test']
CLASS_NAMES = ['Normal', 'Outer Race Fault', 'Inner Race Fault']  # this is a classification problem. Which class is our data set?

print(f'Number of training samples: {len(train_ds)}')
print(f'Number of test samples: {len(test_ds)}')

# display a sample
sample_x, sample_y = test_ds[0]  # should be x: frequency x time, y: label
print(f'Sample shape: {sample_x.shape}  (channels x height x width)')
print(f'Label: {sample_y} ({CLASS_NAMES[sample_y]})')

plt.imshow(sample_x.squeeze(0), origin='lower')  # squeeze removes a dimension so it can be plotted
# shazam works this way to predict songs via 2d forier transform


# %%
BATCH_SIZE   = 64
train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True,  num_workers=0)
test_loader   = DataLoader(test_ds,   batch_size=BATCH_SIZE, shuffle=False, num_workers=0)



# %%

# ------- Build CNN --------

# Input (1 × 64 × 64)
#   └── Conv2d(1→16, 3×3)  + ReLU + MaxPool2d(2×2)   →  16 × 32 × 32  # 3x3 filter  # 2x2 filter
#   └── Conv2d(16→32, 3×3) + ReLU + MaxPool2d(2×2)   →  32 × 16 × 16
#   └── Conv2d(32→64, 3×3) + ReLU + MaxPool2d(2×2)   →  64 × 8 × 8
#   └── Flatten                                        →  4096
#   └── Linear(4096 → 128) + ReLU
#   └── Linear(128 → 3)                               →  logits for 3 classes

# HP: out_channel, kernel size, 
# equation generates padding value

class BearingCNN(nn.Module):
    def __init__(self,):
        super(BearingCNN, self).__init__()
        # --- Feature Extractor ---
        #  Each block: Conv -> BatchNorm -> ReLU -> MaxPool
        self.features = nn.Sequential(
            # Block 1: 1x64x64 -> 13x32x32
            nn.Conv2d(in_channels=1, out_channels=16, kernel_size=3, padding=1),  # stride defaults to 1
            nn.BatchNorm2d(16),  # number of channels
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2), # default stride=kernel_size

            # Block 2: 16x16x16 -> 32x16x16
            nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),

            # Block 3: 32x16x16 -> 64x8x8
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            # num features: 4096
        )
        self.classifier = nn.Sequential(
            # Flatten will take all the numbers in the channels and make it one vector
            # These become features
            nn.Linear(in_features=4096, out_features=128),
            nn.ReLU(),
            nn.Linear(128, 3)  # 3 classes
        )

    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x


model = BearingCNN().to(device)
print(model)

# Count parameters
trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
print(f'Trainable parameters: {trainable_params:,}')

# Sanity check: forward pass with a dummy batch
# This is another way to check that my model runs at the right shapes
dummy = torch.randn(12, 1, 64, 64).to(device)  # batch, channels, size of dimension, size of dimension
out = model(dummy)
print(f'\nOutput shape: {out.shape}')  # should be [12, 3]



# %%
# --------- Training -------------

def train(model, loader, optimizer, lossfn):
    model.train()
    total_loss, correct, n = 0.0, 0, 0

    for X, y in loader:
        X, y = X.to(device), y.to(device)

        optimizer.zero_grad()
        logits = model(X)
        loss = lossfn(logits, y)
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * len(y)
        correct += (logits.argmax(dim=1) == y).sum().item()
        n += len(y)

    return total_loss / n, correct / n


@torch.no_grad()
def evaluate(model, loader, lossfn):
    model.eval()
    total_loss, correct, n = 0.0, 0, 0

    for X, y in loader:
        X, y = X.to(device), y.to(device)
        logits = model(X)
        loss = lossfn(logits, y)

        total_loss += loss.item() * len(y)
        correct += (logits.argmax(dim=1) == y).sum().item()
        n += len(y)

    return total_loss / n, correct / n


# --- Hyperparameters ---
EPOCHS    = 15
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.3)

# --- Training Loop ---
history = {'train_loss': [], 'test_loss': [], 'train_acc': [], 'test_acc': []}

print(f'Training for {EPOCHS} epochs on {device}...')
print(f'{"Epoch":>6}  {"Train Loss":>12}  {"Train Acc":>10}  {"test Loss":>10}  {"test Acc":>8}')
print('-' * 60)

for epoch in range(1, EPOCHS + 1):
    train_loss, train_acc = train(model, train_loader, optimizer, criterion)
    test_loss,   test_acc   = evaluate(model, test_loader, criterion)
    scheduler.step()

    history['train_loss'].append(train_loss)
    history['test_loss'].append(test_loss)
    history['train_acc'].append(train_acc)
    history['test_acc'].append(test_acc)

    print(f'{epoch:>6}  {train_loss:>12.4f}  {train_acc:>9.1%}  {test_loss:>10.4f}  {test_acc:>7.1%}')

print('\n✅ Training complete!')
# %%
