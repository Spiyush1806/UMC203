#==================================================================================================PART1
import json
import numpy as np
import time
from tqdm import tqdm
import cvxopt
import matplotlib.pyplot as plt

# Function to read JSON file
def read_json_file(file_name):
    with open(file_name, 'r') as f:
        data = json.load(f)
    return data

# Load data
data = read_json_file('q1_data.json')

X_train = np.array([np.array(sample[0]) for sample in data['train_data']])
y_train = np.array([sample[1] for sample in data['train_data']])
X_test = np.array([np.array(sample[0]) for sample in data['test_data']])
y_test = np.array([sample[1] for sample in data['test_data']])
print(X_train.shape)

# Perceptron algorithm implementation
def perceptron_algorithm(X_train, y_train, max_iter=10000):
    intermediate_weights = {}
    n_samples, n_features = len(X_train), len(X_train[0])
    w = np.zeros(n_features + 1)

    for epoch in tqdm(range(1, max_iter + 1)):
        change = False  
        for i in range(n_samples):
            x = np.array(X_train[i].tolist() + [1])  # Add bias term
            y = y_train[i]
            if y * np.dot(w, x) <= 0:
                w += y * x
                change = True  
        
        # Store weights at every iteration
        intermediate_weights[epoch] = w.copy()
        
        if not change:
            break  # Stop if no weight updates

    return w, intermediate_weights

# Function to compute misclassification loss
def missclassification_loss(w):
    n_samples = len(X_test)
    loss = 0
    for i in range(n_samples):
        x = np.array(X_test[i].tolist() + [1])
        y = y_test[i]
        if y * np.dot(w, x) <= 0:
            loss += 1
    return loss / n_samples

# Run perceptron algorithm
w, intermediate_weights = perceptron_algorithm(X_train, y_train, max_iter=10000)

# Compute misclassification loss for stored weights
iterations = list(intermediate_weights.keys())
missclassification_losses = [missclassification_loss(w) for w in intermediate_weights.values()]

# Select every 20th iteration for plotting
filtered_iterations = iterations[::20]
filtered_losses = missclassification_losses[::20]

# Plot misclassification loss vs iterations
# Plot misclassification loss vs iterations
plt.figure(figsize=(8, 5))
plt.plot(filtered_iterations, filtered_losses, marker='o', linestyle='-', color='b', label='Misclassification Loss', linewidth=0.3)
plt.xlabel('Iterations')
plt.ylabel('Misclassification Loss')
plt.title('Misclassification Loss vs Iterations')
plt.legend()
plt.grid(True)
plt.show()

#=====================================================================================PART2

start_primal = time.time()
n_samples = 1000
n_features = 27
C=1

P = np.block([
    [np.eye(n_features), np.zeros((n_features, 1 + n_samples))],
    [np.zeros((1 + n_samples, n_features + 1 + n_samples))]  
])
P = cvxopt.matrix(P)

q = np.hstack([np.zeros(n_features + 1), C * np.ones(n_samples)])
q = cvxopt.matrix(q)

# Construct G and h manually
G = np.zeros((2 * n_samples, n_features + 1 + n_samples))
h = np.zeros(2 * n_samples)

# First set of constraints: -y_i (w^T x_i + b) + ξ_i ≤ -1
for i in range(n_samples):
    G[i, :n_features] = -y_train[i] * X_train[i]  # -y_i * X_i
    G[i, n_features] = -y_train[i]  # -y_i * b
    G[i, n_features + 1 + i] = -1  # -ξ_i
    h[i] = -1  # Right-hand side: -1

# Second set of constraints: ξ_i ≥ 0
for i in range(n_samples):
    G[n_samples + i, n_features + 1 + i] = -1  # -ξ_i
    h[n_samples + i] = 0  # Right-hand side: 0

G = cvxopt.matrix(G)
h = cvxopt.matrix(h)

cvxopt.solvers.options['show_progress'] = True
solution = cvxopt.solvers.qp(P, q, G, h)

end_primal = time.time()

# Extract optimal values of w, b
w_opt = np.array(solution['x'][:n_features]).flatten()
b_opt = solution['x'][n_features]

print(f'Optimal w: {w_opt}')
print(f'Optimal b: {b_opt}')
print(f'Primal SVM Time: {end_primal - start_primal:.4f} seconds')

# Given data
n_samples = 1000
n_features = 27
C = 1  # Regularization parameter

# Start timing
start_dual = time.time()

# Compute Gram matrix P (n_samples x n_samples)
Y = y_train[:, None]  # Convert to column vector
P = (Y @ Y.T) * (X_train @ X_train.T)
P = cvxopt.matrix(P)

# q vector (-1 for all terms)
q = cvxopt.matrix(-np.ones(n_samples))

# G and h enforce 0 <= λ <= C
G = np.vstack((-np.eye(n_samples), np.eye(n_samples)))
h = np.hstack((np.zeros(n_samples), C * np.ones(n_samples)))
G = cvxopt.matrix(G)
h = cvxopt.matrix(h)

# A and b enforce sum(λ_i * y_i) = 0
A = cvxopt.matrix(y_train, (1, n_samples), tc='d')
b = cvxopt.matrix(0.0)

# Solve the quadratic program
cvxopt.solvers.options['show_progress'] = True
solution = cvxopt.solvers.qp(P, q, G, h, A, b)

# Extract optimal λ values
lambda_opt = np.array(solution['x']).flatten()

# Compute the optimal weight vector w
w_opt = np.sum(lambda_opt[:, None] * y_train[:, None] * X_train, axis=0)

# Compute the optimal bias b
support_indices = np.where((lambda_opt > 1e-5) & (lambda_opt < C - 1e-5))[0]
b_opt = np.mean(y_train[support_indices] - np.dot(X_train[support_indices], w_opt))

# Stop timing
end_dual = time.time()

# Print optimal values
print(f"Optimal λ: {lambda_opt}")
print(f"Optimal w: {w_opt}")
print(f"Optimal b: {b_opt}")
print(f"Dual SVM time: {end_dual-start_dual:.4f} seconds")

# Compute ξ_i for each sample
margin_violations = 1 - y_train * (X_train @ w_opt + b_opt)
xi = np.maximum(0, margin_violations)

# Identify misclassified points (ξ_i > 1)
misclassified_indices = np.where(xi > 1)[0]

# Identify points inside the margin (0 < ξ_i ≤ 1)
inside_margin_indices = np.where((xi > 0) & (xi <= 1))[0]

# Print results
print(f"Total misclassified points (ξ_i > 1): {len(misclassified_indices)}")
# print(f"Total samples inside margin (0 < ξ_i ≤ 1): {len(inside_margin_indices)}")

misclassified_indices_list=misclassified_indices.tolist()
print('misclassified_indeices_list:',misclassified_indices_list)

X_train_filtered=[]
y_train_filtered=[]
for index in range(1000):
    if index not in misclassified_indices_list:
        X_train_filtered.append(X_train[index])
        y_train_filtered.append(y_train[index])

X_train_filtered=np.array(X_train_filtered)
y_train_filtered=np.array(y_train_filtered)

import numpy as np
import random
from tqdm import tqdm

def perceptron_2(X_train, y_train, max_iter=1000):
    n_samples, n_features = X_train.shape
    w = np.zeros(n_features + 1)
    misclassification_errors = {}
    for _ in tqdm(range(1, max_iter + 1)):
        change = False
        for i in range(n_samples):
            x = np.array(X_train[i].tolist() + [1])
            y = y_train[i]
            if y * np.dot(w, x) <= 0:
                w += y * x
                change = True
                misclassification_errors[_] = missclassification_loss(w)
        if not change:
            print('Algorithm coonverged in iterations:',_)
            break
    return w, misclassification_errors

# Train the perceptron and get misclassification errors
w, misclassification_errors = perceptron_2(X_train_filtered, y_train_filtered, max_iter=1000)

# Plot misclassification loss vs iteration

iterations = list(misclassification_errors.keys())
losses = list(misclassification_errors.values())

plt.figure(figsize=(8, 5))
plt.plot(iterations, losses, marker='o', linestyle='-', color='b', label='Misclassification Loss', linewidth=0.3)
plt.xlabel('Iterations')
plt.ylabel('Misclassification Loss')
plt.title('Misclassification Loss vs Iterations')
plt.legend()
plt.grid(True)
plt.show()


print('===========================================')

import numpy as np
import cvxopt

def rbf_kernel(X1, X2, gamma):
    """Compute the RBF (Gaussian) kernel matrix."""
    K = np.zeros((len(X1), len(X2)))
    for i in range(len(X1)):
        for j in range(len(X2)):
            K[i, j] = np.exp(-gamma * np.linalg.norm(X1[i] - X2[j]) ** 2)
    return K

def train_svm_rbf(X_train, y_train, C, gamma):
    """Train an SVM with the RBF Kernel using Quadratic Programming."""
    n_samples = len(X_train)
    
    # Compute the kernel matrix
    K = rbf_kernel(X_train, X_train, gamma)

    # Construct the quadratic optimization problem
    P = cvxopt.matrix(np.outer(y_train, y_train) * K)
    q = cvxopt.matrix(-np.ones(n_samples))

    G = cvxopt.matrix(np.vstack((-np.eye(n_samples), np.eye(n_samples))))
    h = cvxopt.matrix(np.hstack((np.zeros(n_samples), C * np.ones(n_samples))))

    A = cvxopt.matrix(y_train.astype(float), (1, n_samples), tc='d')
    b = cvxopt.matrix(0.0)

    # Solve QP problem
    cvxopt.solvers.options['show_progress'] = False  # Disable verbose output
    solution = cvxopt.solvers.qp(P, q, G, h, A, b)

    # Extract Lagrange multipliers
    lambda_opt = np.ravel(solution['x'])

    # Find support vectors
    sv_indices = lambda_opt > 1e-5
    support_vectors = X_train[sv_indices]
    support_labels = y_train[sv_indices]
    support_lambdas = lambda_opt[sv_indices]

    return support_vectors, support_labels, support_lambdas, lambda_opt

def compute_bias(support_vectors, support_labels, support_lambdas, gamma):
    """Compute the bias term b using support vectors."""
    K_sv = rbf_kernel(support_vectors, support_vectors, gamma)
    b_values = support_labels - np.sum(support_lambdas[:, None] * support_labels[:, None] * K_sv, axis=0)
    return np.mean(b_values)

def predict(X_test, support_vectors, support_labels, support_lambdas, b_opt, gamma):
    """Predict labels for the given test data."""
    K_test = rbf_kernel(X_test, support_vectors, gamma)
    decision_values = np.dot(K_test, support_lambdas * support_labels) + b_opt
    return np.sign(decision_values)

# Define hyperparameter search space
C_values = [100]
gamma_values = [5]


best_accuracy = 0
best_C, best_gamma = None, None

# Grid search for best C and gamma
for C in tqdm(C_values, desc="Tuning C", position=0):
    for gamma in gamma_values:
        # Train SVM
        support_vectors, support_labels, support_lambdas, lambda_opt = train_svm_rbf(X_train, y_train, C, gamma)

        # Compute bias
        b_opt = compute_bias(support_vectors, support_labels, support_lambdas, gamma)

        # Predict on training data
        predictions = predict(X_train, support_vectors, support_labels, support_lambdas, b_opt, gamma)

        # Compute accuracy
        accuracy = np.mean(predictions == y_train)

        print(f"C={C}, gamma={gamma} -> Training Accuracy: {accuracy * 100:.2f}%")

        # Update best parameters
        if accuracy > best_accuracy:
            best_accuracy = accuracy
            best_C, best_gamma = C, gamma

# Print best hyperparameters
print("\nBest Hyperparameters:")
print(f"C={best_C}, gamma={best_gamma}, Best Accuracy: {best_accuracy * 100:.2f}%")

# Misclassification loss of the best model
support_vectors, support_labels, support_lambdas, lambda_opt = train_svm_rbf(X_train, y_train, best_C, best_gamma)
b_opt = compute_bias(support_vectors, support_labels, support_lambdas, best_gamma)

# Predict on test data
predictions = predict(X_test, support_vectors, support_labels, support_lambdas, b_opt, best_gamma)

# Compute misclassification loss
misclassification_loss = np.mean(predictions != y_test)
print(f"Test Misclassification Loss: {misclassification_loss * 100:.2f}%")


import os
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from PIL import Image  # Pillow for image handling
from torchvision import transforms, datasets
from torchvision.datasets import ImageFolder
from torch.utils.data import random_split, DataLoader, Dataset
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (confusion_matrix, classification_report,
                             roc_auc_score, roc_curve, auc, accuracy_score, recall_score, f1_score, precision_score)
from sklearn.preprocessing import label_binarize
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split
import itertools
import pandas as pd
import seaborn as sns

def q2_get_mnist_jpg_subset(srn: int):
    if os.path.exists("q2_data"):
        return
    else:
        zip_path = "./q2_data.zip"
        def download_zip(srn, zip_path):
            # Placeholder implementation: Replace with actual download logic
            print(f"Downloading data for SRN {srn} to {zip_path}")
            # Example: Simulate creating a zip file
            with open(zip_path, 'w') as f:
                f.write("Simulated zip content")
        
        def download_zip(srn, zip_path):
            # Placeholder implementation: Replace with actual download logic
            print(f"Downloading data for SRN {srn} to {zip_path}")
            # Example: Simulate creating a zip file
            with open(zip_path, 'w') as f:
                f.write("Simulated zip content")
        
        def download_zip(srn, zip_path):
            # Placeholder implementation: Replace with actual download logic
            print(f"Downloading data for SRN {srn} to {zip_path}")
            # Example: Simulate creating a zip file
            with open(zip_path, 'w') as f:
                f.write("Simulated zip content")
        
        download_zip(srn, zip_path)
        def extract_zip(zip_path, extract_to):
            import zipfile
            with zipfile.ZipFile(zip_path, 'r') as zip_ref:
                zip_ref.extractall(extract_to)
        
        extract_zip(zip_path, "q2_data")
# Custom dataset class using PIL for manual image loading
def load_data(batch_size=64):
    transform = transforms.Compose([transforms.Grayscale(), transforms.ToTensor()])
    dataset = ImageFolder(root="q2_data", transform=transform)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    return dataset, dataloader

def train_mlp(train_loader, input_dim, num_classes=10, epochs=1, lr=0.001):
    class MLP(nn.Module):
        def __init__(self, input_dim, num_classes):
            super(MLP, self).__init__()
            self.fc = nn.Linear(input_dim, num_classes)

        def forward(self, x):
            x = self.fc(x)
            return x

    model = MLP(input_dim, num_classes)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)

    for epoch in range(epochs):
        model.train()
        total_loss = 0.0
        correct = 0
        total = 0

        for images, labels in train_loader:
            images = images.view(images.size(0), -1)
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            _, predicted = outputs.max(1)
            correct += predicted.eq(labels).sum().item()
            total += labels.size(0)

        print(f"Epoch {epoch+1}/{epochs}, Loss: {total_loss/len(train_loader):.4f}, Accuracy: {100*correct/total:.2f}%")

    
            
def main():
    q2_get_mnist_jpg_subset(23801)  # Fetch dataset
    dataset, train_loader = load_data()
    input_dim = 28 * 28
    mlp_model = train_mlp(train_loader, input_dim)
    print("MLP Training Completed")

if __name__ == "__main__":
    main()


# =============================
# 2. Convolution Neural Network: 
# =============================
# CNN Model
class CNN(nn.Module):
    def __init__(self, num_classes=10):
        super(CNN, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.fc1 = nn.Linear(64 * 7 * 7, 128)
        self.fc2 = nn.Linear(128, num_classes)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 64 * 7 * 7)  # Flatten
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x

# Load Dataset & Create Data Loaders
def load_data(batch_size=64):
    transform = transforms.Compose([
        transforms.Grayscale(num_output_channels=1),
        transforms.Resize((28, 28)), 
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])

    dataset = datasets.ImageFolder(root="q2_data", transform=transform)
    
    # 80-20 split for train and validation
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=torch.cuda.is_available())
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=torch.cuda.is_available())

    return train_loader, val_loader

from tqdm import tqdm
# Train the CNN Model
def train_cnn():
    srn = 23801
    q2_get_mnist_jpg_subset(srn)  # Ensure dataset is downloaded
    
    print(f"Training CNN on MNIST-JPG subset for SRN: {srn}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = CNN().to(device)

    train_loader, val_loader = load_data()
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    epochs = 10
    for epoch in tqdm(range(epochs)):
        model.train()
        total_loss = 0.0
        correct = 0
        total = 0

        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            _, predicted = outputs.max(1)
            correct += predicted.eq(labels).sum().item()
            total += labels.size(0)

        train_acc = 100 * correct / total
        train_loss = total_loss / len(train_loader)

        # Validation phase
        model.eval()
        val_loss = 0.0
        val_correct = 0
        val_total = 0
        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                loss = criterion(outputs, labels)

                val_loss += loss.item()
                _, predicted = outputs.max(1)
                val_correct += predicted.eq(labels).sum().item()
                val_total += labels.size(0)

        val_acc = 100 * val_correct / val_total
        val_loss /= len(val_loader)

        print(f"Epoch {epoch+1}/{epochs}")
        print(f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%")
        print(f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}%\n")

    torch.save(model.state_dict(), "cnn_mnist.pth")
    print("Model trained and saved as cnn_mnist.pth")

# Run training
train_cnn()

# =============================
# 3. PCA
# =============================
# Load dataset and preprocess
def load_mnist_jpg_data(root_dir="q2_data", image_size=(28, 28)):
    """
    Loads images from subfolders (0 to 9) in `root_dir`.
    Converts each image to grayscale, resizes to `image_size`, and flattens to 1D.

    Returns:
        X (np.ndarray): shape (N, 784) - flattened images
        y (np.ndarray): shape (N,) - digit labels [0..9]
    """
    X_list = []
    y_list = []

    for label_str in sorted(os.listdir(root_dir)):
        label_dir = os.path.join(root_dir, label_str)
        if not os.path.isdir(label_dir):
            continue
        
        label = int(label_str)  # Convert folder name to int
        for img_file in os.listdir(label_dir):
            if img_file.lower().endswith(('.jpg', '.jpeg', '.png')):
                img_path = os.path.join(label_dir, img_file)
                
                # Load, convert to grayscale, resize, and normalize
                with Image.open(img_path).convert('L') as img:
                    img = img.resize(image_size)
                    img_arr = np.array(img, dtype=np.float32) / 255.0
                    img_flat = img_arr.flatten()  # Flatten to 784-dimensional vector

                    X_list.append(img_flat)
                    y_list.append(label)

    X = np.stack(X_list, axis=0)  # shape: (N, 784)
    y = np.array(y_list, dtype=np.int64)  # Labels (N,)

    return X, y

# Train PCA on MNIST-JPG dataset
def train_pca():
    srn = 23801
    q2_get_mnist_jpg_subset(srn)  # Ensure dataset is downloaded

    # Load dataset
    X, y = load_mnist_jpg_data(root_dir="q2_data")

    # Split into training and test sets
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=42
    )

    print(f"Training set size: {X_train.shape}, Test set size: {X_test.shape}")

    # Apply PCA with 100 components
    pca = PCA(n_components=100)
    X_train_pca = pca.fit_transform(X_train)
    X_test_pca = pca.transform(X_test)

    # Print explained variance ratio
    explained_variance = np.sum(pca.explained_variance_ratio_)
    print(f"PCA retains {explained_variance:.2f} of the variance.")

    print(f"Original feature size: {X_train.shape[1]}")
    print(f"Reduced feature size: {X_train_pca.shape[1]}")

    return X_train_pca, X_test_pca, y_train, y_test

# Run PCA training
train_pca()

# =============================
# 4. MLP with PCA
# =============================
# Load dataset and apply PCA
def load_data(batch_size=64, pca_components=50):
    transform = transforms.Compose([
        transforms.Grayscale(num_output_channels=1),
        transforms.Resize((28, 28)),
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])
    dataset = ImageFolder(root="q2_data", transform=transform)

    # Convert dataset into numpy array for PCA
    data = []
    labels = []
    for img, label in dataset:
        data.append(img.numpy().flatten())  # Convert 28x28 to 1D array
        labels.append(label)

    data = np.array(data)
    labels = np.array(labels)

    # Apply PCA
    pca = PCA(n_components=pca_components)
    data_pca = pca.fit_transform(data)

    # Convert back to PyTorch tensors
    data_pca = torch.tensor(data_pca, dtype=torch.float32)
    labels = torch.tensor(labels, dtype=torch.long)

    # Create Dataset & DataLoader
    dataset_pca = torch.utils.data.TensorDataset(data_pca, labels)
    
    # Train-validation split (80-20)
    train_size = int(0.8 * len(dataset_pca))
    val_size = len(dataset_pca) - train_size
    train_dataset, val_dataset = random_split(dataset_pca, [train_size, val_size])

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    return train_loader, val_loader, pca

# Define MLP Model
class MLP(nn.Module):
    def __init__(self, input_size=50, hidden_size=128, num_classes=10):
        super(MLP, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, num_classes)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x

# Train MLP Model
def train_mlp(train_loader, val_loader, input_size=50, epochs=10, lr=0.001):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = MLP(input_size=input_size).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)

    for epoch in range(epochs):
        model.train()
        total_loss = 0.0
        correct = 0
        total = 0

        for features, labels in train_loader:
            features, labels = features.to(device), labels.to(device)
            
            optimizer.zero_grad()
            outputs = model(features)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            _, predicted = outputs.max(1)
            correct += predicted.eq(labels).sum().item()
            total += labels.size(0)

        train_acc = 100 * correct / total
        train_loss = total_loss / len(train_loader)

        # Validation phase
        model.eval()
        val_loss = 0.0
        val_correct = 0
        val_total = 0
        with torch.no_grad():
            for features, labels in val_loader:
                features, labels = features.to(device), labels.to(device)
                outputs = model(features)
                loss = criterion(outputs, labels)

                val_loss += loss.item()
                _, predicted = outputs.max(1)
                val_correct += predicted.eq(labels).sum().item()
                val_total += labels.size(0)

        val_acc = 100 * val_correct / val_total
        val_loss /= len(val_loader)

        print(f"Epoch {epoch+1}/{epochs}")
        print(f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%")
        print(f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}%\n")

    return model

# Main function to train MLP with PCA
def main():
    q2_get_mnist_jpg_subset(23801)  # Fetch dataset
    train_loader, val_loader, pca = load_data(pca_components=50)  # Use PCA with 50 components
    mlp_model = train_mlp(train_loader, val_loader, input_size=50)
    print("MLP Training with PCA Completed")
    torch.save(mlp_model.state_dict(), "mlp_pca_mnist.pth")
    print("Model saved as mlp_pca_mnist.pth")

if __name__ == "__main__":
    main()

# =============================
# 5. Logistic Regression with PCA:
# =============================
# Load dataset and preprocess images
def load_mnist_jpg_data(root_dir="q2_data", image_size=(28, 28)):
    X_list, y_list = [], []
    
    for label_str in sorted(os.listdir(root_dir)):
        label_dir = os.path.join(root_dir, label_str)
        if not os.path.isdir(label_dir):
            continue
        
        label = int(label_str)  # Convert folder name to integer class label
        for img_file in os.listdir(label_dir):
            if img_file.lower().endswith(('.jpg', '.jpeg', '.png')):
                img_path = os.path.join(label_dir, img_file)
                
                # Load, convert to grayscale, resize, and normalize
                with Image.open(img_path).convert('L') as img:
                    img = img.resize(image_size)
                    img_arr = np.array(img, dtype=np.float32) / 255.0
                    img_flat = img_arr.flatten()  # Flatten to 784-dimensional vector

                    X_list.append(img_flat)
                    y_list.append(label)

    X = np.stack(X_list, axis=0)  # shape: (N, 784)
    y = np.array(y_list, dtype=np.int64)  # Labels (N,)

    return X, y

# Train PCA and apply Logistic Regression
def train_logistic_regression_with_pca():
    srn = 23801
    q2_get_mnist_jpg_subset(srn)  # Ensure dataset is downloaded

    print(f"Training Logistic Regression with PCA on MNIST-JPG subset for SRN: {srn}")

    # Load dataset
    X, y = load_mnist_jpg_data(root_dir="q2_data")

    # Split into training and test sets (80-20)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=42
    )

    # Apply PCA with 100 components
    pca = PCA(n_components=100)
    X_train_pca = pca.fit_transform(X_train)
    X_test_pca = pca.transform(X_test)

    print(f"PCA Explained Variance: {np.sum(pca.explained_variance_ratio_):.2f}")

    # Multi-class Logistic Regression
    clf = LogisticRegression(max_iter=1000, solver="lbfgs", multi_class="multinomial")
    clf.fit(X_train_pca, y_train)
    y_pred = clf.predict(X_test_pca)

    print("\n=== Multi-class Logistic Regression ===")
    print("Accuracy:", accuracy_score(y_test, y_pred))
    print(classification_report(y_test, y_pred))

    # One-vs-Rest (OvR) Binary Classification for each class
    y_train_bin = label_binarize(y_train, classes=np.arange(10))
    y_test_bin = label_binarize(y_test, classes=np.arange(10))

    ovr_classifiers = {}
    for i in range(10):
        ovr_clf = LogisticRegression(max_iter=1000, solver="lbfgs")
        ovr_clf.fit(X_train_pca, y_train_bin[:, i])
        ovr_classifiers[i] = ovr_clf

        # Predict binary labels
        y_pred_ovr = ovr_clf.predict(X_test_pca)
        print(f"\n=== One-vs-Rest Classifier for Class {i} ===")
        print("Accuracy:", accuracy_score(y_test_bin[:, i], y_pred_ovr))

# Run Logistic Regression with PCA
train_logistic_regression_with_pca()


# =============================
#  Deliverables: 1.
# =============================
# Function to load dataset
def load_mnist_jpg_data(root_dir="q2_data", image_size=(28, 28)):
    X_list = []
    y_list = []

    for label_str in sorted(os.listdir(root_dir)):
        label_dir = os.path.join(root_dir, label_str)
        if not os.path.isdir(label_dir):
            continue
        
        label = int(label_str)  
        for img_file in os.listdir(label_dir):
            if img_file.lower().endswith(('.jpg', '.jpeg', '.png')):
                img_path = os.path.join(label_dir, img_file)
                
                with Image.open(img_path).convert('L') as img:
                    img = img.resize(image_size)
                    img_arr = np.array(img, dtype=np.float32) / 255.0  
                    img_flat = img_arr.flatten()  

                    X_list.append(img_flat)
                    y_list.append(label)

    X = np.stack(X_list, axis=0)  
    y = np.array(y_list, dtype=np.int64)  

    return X, y

# Function to reconstruct an image using PCA
def reconstruct_image(image_index=0):
    """
    Reconstructs an image using different numbers of principal components.
    
    Parameters:
        image_index (int): Index of the image to reconstruct (default: first image).
    """
    # Fetch dataset if not present
    srn = 23801
    q2_get_mnist_jpg_subset(srn)

    # Load dataset
    X, y = load_mnist_jpg_data(root_dir="q2_data")

    # Ensure valid image index
    if image_index >= len(X):
        print(f"Invalid image index! Must be between 0 and {len(X) - 1}.")
        return

    # Select the specified image
    original_image = X[image_index].reshape(28, 28)

    # Apply PCA on the entire dataset
    pca = PCA(n_components=784)  # Full PCA for reconstruction
    X_pca = pca.fit_transform(X)

    # Principal components to use for reconstruction
    components_list = [1, 2, 3, 5, 10, 20, 50, 100, 200, 784]

    fig, axes = plt.subplots(2, 5, figsize=(12, 5))

    for i, components in enumerate(components_list):
        # Reduce to selected number of components
        pca_reduced = PCA(n_components=components)
        X_reduced = pca_reduced.fit_transform(X)  
        X_reconstructed = pca_reduced.inverse_transform(X_reduced)  

        # Get the reconstructed image
        reconstructed_image = X_reconstructed[image_index].reshape(28, 28)  

        ax = axes[i // 5, i % 5]
        ax.imshow(reconstructed_image, cmap="gray")
        ax.set_title(f"{components} components")
        ax.axis("off")

    # Display the original image
    plt.figure(figsize=(3, 3))
    plt.imshow(original_image, cmap="gray")
    plt.title("Original Image")
    plt.axis("off")

    plt.suptitle(f"Reconstruction of Image Index {image_index} with Different PCA Components")
    plt.show()

# Run image reconstruction (Change the index to select different images)
reconstruct_image(image_index=0)

# =============================         
#  Deliverables: 2. 
# =============================
# Load and preprocess dataset
def load_mnist_jpg_data(root_dir="q2_data", image_size=(28, 28)):
    X_list = []
    y_list = []

    for label_str in sorted(os.listdir(root_dir)):
        label_dir = os.path.join(root_dir, label_str)
        if not os.path.isdir(label_dir):
            continue
        
        label = int(label_str)  
        for img_file in os.listdir(label_dir):
            if img_file.lower().endswith(('.jpg', '.jpeg', '.png')):
                img_path = os.path.join(label_dir, img_file)
                
                with Image.open(img_path).convert('L') as img:
                    img = img.resize(image_size)
                    img_arr = np.array(img, dtype=np.float32) / 255.0  
                    img_flat = img_arr.flatten()  

                    X_list.append(img_flat)
                    y_list.append(label)

    X = np.stack(X_list, axis=0)  
    y = np.array(y_list, dtype=np.int64)  

    return X, y

# Train PCA and Logistic Regression (multi-class)
def train_and_evaluate_models():
    # Fetch dataset if not present
    srn = 23801
    q2_get_mnist_jpg_subset(srn)

    # Load dataset
    X, y = load_mnist_jpg_data(root_dir="q2_data")

    # Split dataset
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=42
    )

    # Apply PCA
    pca = PCA(n_components=100)
    X_train_pca = pca.fit_transform(X_train)
    X_test_pca = pca.transform(X_test)

    # Train Multi-Class Logistic Regression (One-vs-All)
    clf_multiclass = LogisticRegression(multi_class="multinomial", solver="lbfgs", max_iter=500)
    clf_multiclass.fit(X_train_pca, y_train)
    y_pred_multi = clf_multiclass.predict(X_test_pca)

    # Train One-vs-Rest Logistic Regression (10 binary classifiers)
    clf_ovr = LogisticRegression(multi_class="ovr", solver="liblinear", max_iter=500)
    clf_ovr.fit(X_train_pca, y_train)
    y_pred_ovr = clf_ovr.predict(X_test_pca)

    # Compute confusion matrices
    cm_multiclass = confusion_matrix(y_test, y_pred_multi)
    cm_ovr = confusion_matrix(y_test, y_pred_ovr)

    # Function to compute and display evaluation metrics
    def evaluate_model(y_true, y_pred, model_name):
        accuracy = accuracy_score(y_true, y_pred)
        precision = precision_score(y_true, y_pred, average=None)
        recall = recall_score(y_true, y_pred, average=None)
        f1 = f1_score(y_true, y_pred, average=None)

        print(f"\n===== {model_name} Metrics =====")
        print(f"Overall Accuracy: {accuracy:.4f}")
        
        metrics_df = pd.DataFrame({
            "Class": np.arange(10),
            "Precision": precision,
            "Recall": recall,
            "F1-Score": f1
        })

        print(metrics_df.to_string(index=False))

        return accuracy, precision, recall, f1

    # Evaluate both models
    acc_multi, prec_multi, rec_multi, f1_multi = evaluate_model(y_test, y_pred_multi, "Multinomial Logistic Regression")
    acc_ovr, prec_ovr, rec_ovr, f1_ovr = evaluate_model(y_test, y_pred_ovr, "One-vs-Rest Logistic Regression")

    # Plot confusion matrices
    def plot_confusion_matrix(cm, title):
        plt.figure(figsize=(8,6))
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=range(10), yticklabels=range(10))
        plt.xlabel("Predicted Label")
        plt.ylabel("True Label")
        plt.title(title)
        plt.show()

    plot_confusion_matrix(cm_multiclass, "Confusion Matrix: Multinomial Logistic Regression")
    plot_confusion_matrix(cm_ovr, "Confusion Matrix: One-vs-Rest Logistic Regression")

    # Compare Results Across Both Models
    comparison_df = pd.DataFrame({
        "Class": np.arange(10),
        "Multinomial Precision": prec_multi,
        "OvR Precision": prec_ovr,
        "Multinomial Recall": rec_multi,
        "OvR Recall": rec_ovr,
        "Multinomial F1-Score": f1_multi,
        "OvR F1-Score": f1_ovr
    })

    print("\n===== Model Comparison Across Classes =====")
    print(comparison_df.to_string(index=False))

    print("\n===== Overall Model Comparison =====")
    print(f"Multinomial LR - Accuracy: {acc_multi:.4f}")
    print(f"One-vs-Rest LR - Accuracy: {acc_ovr:.4f}")

# Run training and evaluation
train_and_evaluate_models()

# =============================
#  Deliverables: 3.     
# =============================
# Load dataset and preprocess images
def load_mnist_jpg_data(root_dir="q2_data", image_size=(28, 28)):
    X_list, y_list = [], []
    
    for label_str in sorted(os.listdir(root_dir)):
        label_dir = os.path.join(root_dir, label_str)
        if not os.path.isdir(label_dir):
            continue
        
        label = int(label_str)  # Convert folder name to integer class label
        for img_file in os.listdir(label_dir):
            if img_file.lower().endswith(('.jpg', '.jpeg', '.png')):
                img_path = os.path.join(label_dir, img_file)
                
                # Load, convert to grayscale, resize, and normalize
                with Image.open(img_path).convert('L') as img:
                    img = img.resize(image_size)
                    img_arr = np.array(img, dtype=np.float32) / 255.0
                    img_flat = img_arr.flatten()  # Flatten to 784-dimensional vector

                    X_list.append(img_flat)
                    y_list.append(label)

    X = np.stack(X_list, axis=0)  # shape: (N, 784)
    y = np.array(y_list, dtype=np.int64)  # Labels (N,)

    return X, y

# Train PCA and compute AUC score for one-vs-rest logistic regression
def train_and_evaluate_auc():
    srn = 23801
    q2_get_mnist_jpg_subset(srn)  # Ensure dataset is downloaded

    print(f"Training One-vs-Rest Logistic Regression with AUC Computation for SRN: {srn}")

    # Load dataset
    X, y = load_mnist_jpg_data(root_dir="q2_data")

    # Split into training and test sets (80-20)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=42
    )

    # Apply PCA with 100 components
    pca = PCA(n_components=100)
    X_train_pca = pca.fit_transform(X_train)
    X_test_pca = pca.transform(X_test)

    print(f"PCA Explained Variance: {np.sum(pca.explained_variance_ratio_):.2f}")

    # One-vs-Rest (OvR) Binary Classification for each class
    y_train_bin = label_binarize(y_train, classes=np.arange(10))
    y_test_bin = label_binarize(y_test, classes=np.arange(10))

    auc_scores = []
    plt.figure(figsize=(10, 8))

    for i in range(10):
        ovr_clf = LogisticRegression(max_iter=1000, solver="lbfgs")
        ovr_clf.fit(X_train_pca, y_train_bin[:, i])

        # Compute predicted probabilities
        y_prob = ovr_clf.predict_proba(X_test_pca)[:, 1]

        # Compute AUC score
        auc = roc_auc_score(y_test_bin[:, i], y_prob)
        auc_scores.append(auc)

        # Compute ROC curve
        fpr, tpr, _ = roc_curve(y_test_bin[:, i], y_prob)

        # Plot ROC curve
        plt.plot(fpr, tpr, label=f"Class {i} (AUC = {auc:.2f})")

    # Compute average AUC score
    avg_auc = np.mean(auc_scores)
    print(f"\nAverage AUC Score across all classes: {avg_auc:.4f}")

    # Finalize ROC curve plot
    plt.plot([0, 1], [0, 1], linestyle="--", color="gray")  # Random classifier line
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC Curves for One-vs-Rest Logistic Regression")
    plt.legend()
    plt.show()

# Run Logistic Regression with PCA and AUC computation
train_and_evaluate_auc()

#============================================================================== PART 1
import json
import numpy as np

# Load the JSON file
with open("q3_linear_1.json", "r") as file:
    data_1 = json.load(file)
with open("q3_linear_2.json", "r") as file:
    data_2 = json.load(file)

# Extract data from the dictionary
X_train_1 = data_1["X_train"]
y_train_1 = data_1["y_train"]
X_test_1 = data_1["X_test"]
y_test_1 = data_1["y_test"]
X_train_2 = data_2["X_train"]
y_train_2 = data_2["y_train"]
X_test_2 = data_2["X_test"]
y_test_2 = data_2["y_test"]

# Linear regression with MSE loss

X_1=np.array(X_train_1)
Y_1=np.array(y_train_1)
X_2=np.array(X_train_2)
Y_2=np.array(y_train_2)

#Loss functions
def MSE_ols(X, Y, w):
    return np.mean((X @ w - Y) ** 2)

def MSE_rr(X, Y, w, lambda_):
    return np.mean((X @ w - Y) ** 2) + lambda_ * w.T @ w

# w = (X^T X)^(-1) X^T Y
w_ols_1 = np.linalg.inv(X_1.T @ X_1) @ X_1.T @ Y_1
w_ols_2 = np.linalg.inv(X_2.T @ X_2) @ X_2.T @ Y_2

# w = (X^T X + lambda I)^(-1) X^T Y
lambda_ = 1
w_rr_1 = np.linalg.inv(X_1.T @ X_1 + lambda_ * np.eye(X_1.shape[1])) @ X_1.T @ Y_1
w_rr_2 = np.linalg.inv(X_2.T @ X_2 + lambda_ * np.eye(X_2.shape[1])) @ X_2.T @ Y_2

print('MSE loss for OLS on dataset 1:', MSE_ols(X_1, Y_1, w_ols_1))
print('MSE loss for OLS on dataset 2:', MSE_ols(X_2, Y_2, w_ols_2))

print('MSE loss for RR on dataset 1:', MSE_rr(X_1, Y_1, w_rr_1, lambda_))
print('MSE loss for RR on dataset 2:', MSE_rr(X_2, Y_2, w_rr_2, lambda_))

print('w_ols_1:', w_ols_1)
print('w_rr_1:', w_rr_1)

np.savetxt("w_ols_23801.csv", w_ols_2, delimiter = ",")
np.savetxt("w_rr_23801.csv", w_rr_2, delimiter = ",")

#============================================================================== PART 2
import numpy as np
import pandas as pd
from sklearn.svm import SVR
from sklearn.preprocessing import StandardScaler
import matplotlib
matplotlib.use('TkAgg')  # or 'Agg' if no GUI is needed
import matplotlib.pyplot as plt


# 1. Load data
df = pd.read_csv("CHL.csv")
close_prices = df["Close"].to_numpy().reshape(-1, 1)
 
print(close_prices.shape)

# 2. Choose a window size t for your features
t = 7

# 3. Create feature matrix X and target vector y
#    For each day i, the features are the previous t closing prices, the target is the next day close
X, y = [], []
for i in range(t, len(close_prices)):
    X.append(close_prices[i-t:i, 0])
    y.append(close_prices[i, 0])
X, y = np.array(X), np.array(y)

# 4. Split the data into training and test sets
split = int(0.8 * len(X))
X_train, X_test = X[:split], X[split:]
y_train, y_test = y[:split], y[split:]

# 5. Scale the features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# 6. Train the SVR model
model = SVR(kernel='rbf')
model.fit(X_train_scaled, y_train)

# 7. Predict the test set
y_pred = model.predict(X_test_scaled)

# 8. Calculate the moving average
moving_avg = np.convolve(y_test, np.ones(t)/t, mode='valid')

# 9. Plot all three
plt.figure(figsize=(10, 6))
plt.plot(y_test, label='Actual Closing Price', color='blue')
plt.plot(y_pred, label='Predicted Closing Price (SVR)', color='red')
plt.plot(moving_avg, label=f'{t}-Day Moving Average', color='green', linestyle='--')

plt.title('SVR Results vs Actual Price and Moving Average')
plt.xlabel('Time Step (Test Set)')
plt.ylabel('Price')
plt.legend()
plt.grid(True)
plt.show()

