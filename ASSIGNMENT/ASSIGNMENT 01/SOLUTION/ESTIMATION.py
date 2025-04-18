import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), 'oracle'))
import oracle


# Load the dataset
mysrn = 23801
res = oracle.q1_fish_train_test_data(mysrn)
attributes = res[0]
train_images = res[1]
train_labels = res[2]
test_images = res[3]
test_labels = res[4]

# Sample sizes to analyze
sample_sizes = [50, 100, 500, 1000, 2000, 4000]

# Simulated dataset (replace this with your actual dataset)
num_samples = 20000
num_classes = np.unique(train_labels).shape[0]
train_images = np.random.rand(num_samples, 3, 32, 32)  # Shape (20000, 3, 32, 32)
train_labels = np.random.randint(0, num_classes, num_samples)  # Shape (20000,)

# Reshape images to 2D (Flatten each image from (3, 32, 32) → (3072,))
train_images = train_images.reshape(num_samples, -1)  # Shape (20000, 3072)

# Store results
l2_norm_means = {label: [] for label in np.unique(train_labels)}
frobenius_norm_covs = {label: [] for label in np.unique(train_labels)}

# Compute norms for each class at different sample sizes
for n in sample_sizes:
    for label in np.unique(train_labels):
        # Select n samples from this class
        class_samples = train_images[train_labels == label]

        if len(class_samples) < n:
            l2_norm_means[label].append(np.nan)  # Not enough samples
            frobenius_norm_covs[label].append(np.nan)
            continue  

        class_samples = class_samples[:n]  # Use only first n samples

        # Compute Mean Vector and Covariance Matrix
        mean_vector = np.mean(class_samples, axis=0)
        cov_matrix = np.cov(class_samples, rowvar=False)

        # Compute L2 norm and Frobenius norm
        l2_norm_means[label].append(np.linalg.norm(mean_vector, ord=2))
        frobenius_norm_covs[label].append(np.linalg.norm(cov_matrix, ord='fro'))

# Convert results to Pandas DataFrames
df_l2_norm = pd.DataFrame(l2_norm_means, index=sample_sizes)
df_frobenius_norm = pd.DataFrame(frobenius_norm_covs, index=sample_sizes)

# Save results to CSV files
df_l2_norm.to_csv("l2_norm_means.csv")
df_frobenius_norm.to_csv("frobenius_norm_covariances.csv")

# 📊 Plot Results
plt.figure(figsize=(12, 5))

# Plot L2 Norms
plt.subplot(1, 2, 1)
for label in np.unique(train_labels):
    plt.plot(sample_sizes, df_l2_norm[label], marker='o', label=f'Class {label}')
plt.xlabel("Number of Samples (n)")
plt.ylabel("L2 Norm of Mean Vectors")
plt.title("Change in Mean Vector Norms")
plt.legend()

# Plot Frobenius Norms
plt.subplot(1, 2, 2)
for label in np.unique(train_labels):
    plt.plot(sample_sizes, df_frobenius_norm[label], marker='s', label=f'Class {label}')
plt.xlabel("Number of Samples (n)")
plt.ylabel("Frobenius Norm of Covariance Matrices")
plt.title("Change in Covariance Matrix Norms")
plt.legend()

plt.tight_layout()
plt.show()
