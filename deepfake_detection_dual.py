# ✅ Section 1: Imports
import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
from glob import glob

# PyTorch
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader

# TensorFlow
import tensorflow as tf
from tensorflow.keras import layers, models

# ✅ Section 2: Load & Preprocess Dataset
import os
import cv2
import numpy as np
from glob import glob

img_size = 128
max_images = 10000  # Total per category

def load_images(folder, label, max_count):
    data = []
    for i, path in enumerate(glob(os.path.join(folder, '*.jpg'))):
        if i >= max_count:
            break
        img = cv2.imread(path, cv2.IMREAD_COLOR)
        if img is None:
            print(f"Warning: Unable to read image {path}")
            continue
        img = cv2.resize(img, (img_size, img_size))
        img = img.astype(np.float32) / 255.0
        data.append((img, label))
    return data

# ✅ Multiple Real & Fake folders
real_paths = [
    'D:\\DeepFake-Detection\\dataset\\Train\\Real',
    #'D:\\DeepFake-Detection\\dataset\\Train\\Real1',
    #'D:\\DeepFake-Detection\\dataset\\Train\\Real2',
    # Add more if needed
]

fake_paths = [
    'D:\\DeepFake-Detection\\dataset\\Train\\Fake',
   # 'D:\\DeepFake-Detection\\dataset\\Train\\Fake4',
    #'D:\\DeepFake-Detection\\dataset\\Train\\Fake3',
    # Add more if needed
]

# Load real images
real = []
for path in real_paths:
    real += load_images(path, 0, max_images // len(real_paths))

# Load fake images
fake = []
for path in fake_paths:
    fake += load_images(path, 1, max_images // len(fake_paths))

# Combine and shuffle
data = real + fake
np.random.shuffle(data)

# Final arrays
X = np.array([x[0] for x in data], dtype=np.float32)
y = np.array([x[1] for x in data], dtype=np.int32)

# Optional: print stats
print(f"✅ Loaded {len(real)} real images and {len(fake)} fake images. Total: {len(data)} samples")

# ✅ Section 3: PyTorch Implementation
class DeepFakeDataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.tensor(X, dtype=torch.float32).permute(0, 3, 1, 2)  # NCHW format
        self.y = torch.tensor(y, dtype=torch.long)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

class PyTorchCNN(nn.Module):
    def __init__(self):
        super(PyTorchCNN, self).__init__()
        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(32 * 32 * 32, 64)
        self.fc2 = nn.Linear(64, 2)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 32 * 32 * 32)
        x = F.relu(self.fc1(x))
        return self.fc2(x)

# ✅ Section 4: TensorFlow Implementation
def create_tf_model():
    model = models.Sequential([
        layers.Input(shape=(img_size, img_size, 3)),
        layers.Conv2D(16, 3, activation='relu', padding='same'),
        layers.MaxPooling2D(),
        layers.Conv2D(32, 3, activation='relu', padding='same'),
        layers.MaxPooling2D(),
        layers.Flatten(),
        layers.Dense(64, activation='relu'),
        layers.Dense(1, activation='sigmoid')
    ])
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    return model

# ✅ Section 5: Train PyTorch Model (Optional)
from torchvision import transforms

# Data augmentation transforms
transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(10),
    transforms.ToTensor()
])
class AugmentedDeepFakeDataset(Dataset):
    def __init__(self, X, y, transform=None):
        self.X = X
        self.y = y
        self.transform = transform

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        img = self.X[idx]
        label = self.y[idx]
        if self.transform:
            img = self.transform(img)
        else:
            img = torch.tensor(img, dtype=torch.float32).permute(2, 0, 1)
        return img, label

aug_dataset = AugmentedDeepFakeDataset(X, y, transform=transform)
loader = DataLoader(aug_dataset, batch_size=16, shuffle=True)
model_pt = PyTorchCNN()
optimizer = optim.Adam(model_pt.parameters(), lr=0.001)
criterion = nn.CrossEntropyLoss()

num_epochs = 30  # Increased epochs
for epoch in range(num_epochs):
    total_loss = 0
    for inputs, labels in loader:
        optimizer.zero_grad()
        labels = labels.long()  # Convert labels to LongTensor for CrossEntropyLoss
        outputs = model_pt(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    print(f"Epoch {epoch+1}, Loss: {total_loss:.4f}")

# ✅ Section 6: Train TensorFlow Model (Optional)
# model_tf = create_tf_model()
# model_tf.fit(X, y, epochs=5, batch_size=16, validation_split=0.2)

# ✅ Section 7: Predict (Demo)
#def predict_tf(model, image_path):
 #   img = cv2.imread(image_path)
  #  img = cv2.resize(img, (img_size, img_size)).astype(np.float32) / 255.0
   # pred = model.predict(np.expand_dims(img, axis=0))
    #label = "Fake" if pred[0][0] > 0.5 else "Real"
    #print(f"Prediction: {label} ({pred[0][0]:.2f})")

#predict_tf(model_tf, '../dataset/fake/sample.jpg')torch.save(model_pt.state_dict(), "models/pytorch_model.pth")
# Add this after training in your script
torch.save(model_pt.state_dict(), "deepfake_pytorch_model.pth")
print("✅ Model saved as deepfake_pytorch_model.pth")

def predict_image_pt(image_path):
    img = cv2.imread(image_path)
    img = cv2.resize(img, (img_size, img_size)).astype(np.float32) / 255.0
    img_tensor = torch.tensor(img).permute(2, 0, 1).unsqueeze(0).float()

    model_pt.eval()
    with torch.no_grad():
        output = model_pt(img_tensor)
        prob = F.softmax(output, dim=1)
        confidence, pred = torch.max(prob, 1)

    label = "Fake" if pred.item() == 1 else "Real"
    print(f"Prediction: {label} (Confidence: {confidence.item():.2f})")
    return label



# ✅ Section 5.1: Evaluate PyTorch Model Accuracy
def evaluate_model(model, dataset):
    loader = DataLoader(dataset, batch_size=16, shuffle=False)
    model.eval()
    correct = 0
    total = 0

    with torch.no_grad():
        for inputs, labels in loader:
            outputs = model(inputs)
            predicted = torch.argmax(outputs, dim=1)
            correct += (predicted == labels).sum().item()
            total += labels.size(0)

    accuracy = 100 * correct / total
    print(f"✅ Model Accuracy: {accuracy:.2f}%")

# Evaluate on training data (or replace with validation/test set if available)

evaluate_model(model_pt, aug_dataset)

# Add validation/testing code
def load_validation_data():
    val_real_paths = [
        'D:\\DeepFake-Detection\\dataset\\Validation\\Real',
    ]
    val_fake_paths = [
        'D:\\DeepFake-Detection\\dataset\\Validation\\Fake',
    ]

    val_real = []
    for path in val_real_paths:
        val_real += load_images(path, 0, max_images // len(val_real_paths))

    val_fake = []
    for path in val_fake_paths:
        val_fake += load_images(path, 1, max_images // len(val_fake_paths))

    val_data = val_real + val_fake
    np.random.shuffle(val_data)

    X_val = np.array([x[0] for x in val_data], dtype=np.float32)
    y_val = np.array([x[1] for x in val_data], dtype=np.int32)

    return X_val, y_val

X_val, y_val = load_validation_data()
val_dataset = DeepFakeDataset(X_val, y_val)

print(f"✅ Loaded {len(X_val)} validation samples.")

