"""
Day 5, Week 2: PyTorch classifier exercise.

Your ISL project used TensorFlow/Keras — this exercise is deliberately
in PyTorch instead, since that's the framework most GenAI/ML job
descriptions ask for alongside TensorFlow, and the muscle memory doesn't
transfer 1:1 (different API style: explicit training loops instead of
.fit(), explicit device management, nn.Module classes instead of
Sequential/Functional).

Dataset: sklearn's built-in "digits" dataset — 1797 images, 8x8 pixels,
handwritten digits 0-9. Chosen deliberately over MNIST: no download
needed, trains in seconds on CPU, and is small enough to actually watch
every step happen. Same core problem shape as your ISL alphabet
classification (small grayscale images -> a category), just numbers
instead of hand gestures.

Run: python day5_pytorch_classifier.py
"""

import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.datasets import load_digits
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import train_test_split

# ---------------------------------------------------------------------
# 1. Load and prepare data
# ---------------------------------------------------------------------
digits = load_digits()
X, y = digits.data, digits.target  # X: (1797, 64) flattened 8x8 images

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# Normalize to [0, 1] — same reasoning as normalizing your ISL images:
# consistent input scale helps the network train faster and more stably.
X_train = X_train / 16.0
X_test = X_test / 16.0

# Convert to PyTorch tensors. Note the explicit dtype — PyTorch is
# stricter about this than Keras, which quietly casts for you.
X_train_t = torch.tensor(X_train, dtype=torch.float32).reshape(-1, 1, 8, 8)
X_test_t = torch.tensor(X_test, dtype=torch.float32).reshape(-1, 1, 8, 8)
y_train_t = torch.tensor(y_train, dtype=torch.long)
y_test_t = torch.tensor(y_test, dtype=torch.long)


# ---------------------------------------------------------------------
# 2. Define the model — a small CNN, same conv/pool building blocks as
#    your ISL project's Phase 1 CNN, expressed as an explicit nn.Module
#    class instead of Keras' Sequential API.
# ---------------------------------------------------------------------
class DigitCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=16, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(kernel_size=2)
        self.conv2 = nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3, padding=1)
        self.fc1 = nn.Linear(32 * 2 * 2, 64)
        self.fc2 = nn.Linear(64, 10)  # 10 digit classes
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.3)

    def forward(self, x):
        x = self.pool(self.relu(self.conv1(x)))  # 8x8 -> 4x4
        x = self.pool(self.relu(self.conv2(x)))  # 4x4 -> 2x2
        x = x.view(x.size(0), -1)  # flatten
        x = self.dropout(self.relu(self.fc1(x)))
        return self.fc2(x)  # raw logits — CrossEntropyLoss applies softmax internally


model = DigitCNN()
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# ---------------------------------------------------------------------
# 3. Training loop — written out explicitly. This is the biggest
#    Keras-vs-PyTorch difference: Keras' model.fit() hides all of this;
#    PyTorch makes you write the loop, which is exactly why it's worth
#    doing once by hand — you can now explain every step if asked.
# ---------------------------------------------------------------------
EPOCHS = 30
BATCH_SIZE = 32

print("Training...")
for epoch in range(EPOCHS):
    model.train()
    permutation = torch.randperm(X_train_t.size(0))
    epoch_loss = 0.0

    for i in range(0, X_train_t.size(0), BATCH_SIZE):
        indices = permutation[i:i + BATCH_SIZE]
        batch_x, batch_y = X_train_t[indices], y_train_t[indices]

        optimizer.zero_grad()          # PyTorch accumulates gradients by
                                        # default — you must clear them
                                        # each step, Keras does this for you
        outputs = model(batch_x)
        loss = criterion(outputs, batch_y)
        loss.backward()                # compute gradients
        optimizer.step()               # apply the gradient descent step

        epoch_loss += loss.item()

    if (epoch + 1) % 5 == 0:
        print(f"Epoch {epoch + 1}/{EPOCHS} — loss: {epoch_loss:.4f}")

# ---------------------------------------------------------------------
# 4. Evaluation — same metrics vocabulary as your Day 2 revision
#    (precision/recall/F1), now applied to something you trained
#    yourself in a new framework.
# ---------------------------------------------------------------------
model.eval()
with torch.no_grad():  # disables gradient tracking — we're not training now
    predictions = model(X_test_t)
    predicted_labels = torch.argmax(predictions, dim=1)

accuracy = (predicted_labels == y_test_t).float().mean().item()
print(f"\nTest accuracy: {accuracy:.2%}")

print("\nClassification report:")
print(classification_report(y_test, predicted_labels.numpy()))

print("Confusion matrix:")
print(confusion_matrix(y_test, predicted_labels.numpy()))

# ---------------------------------------------------------------------
# Self-check: which digits does this model confuse most often, and does
# that make visual sense (e.g., handwritten 4s and 9s, or 3s and 8s, are
# genuinely easy to confuse)? This is the same confusion-matrix-reading
# skill from your ISL project's letter-confusion analysis — same
# question, different dataset.
# ---------------------------------------------------------------------
