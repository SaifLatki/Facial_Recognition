"""
All project visualizations: samples, predictions, eigenfaces
"""

import matplotlib.pyplot as plt
import numpy as np

def plot_results(X_test, y_test, y_pred, pca, scaler, lfw_images, h, w, metrics, n_samples=6):
    """Main visualization dashboard"""
    fig = plt.figure(figsize=(18, 12))
    
    indices = np.random.choice(len(X_test), n_samples, replace=False)
    
    # 1. Sample test faces (true labels)
    for i, idx in enumerate(indices):
        ax = plt.subplot(2, n_samples, i + 1)
        plt.imshow(lfw_images[idx].reshape(h, w), cmap='gray')
        plt.title(f"True: {'Arnold' if y_test[idx] else 'Other'}", fontsize=10)
        plt.axis('off')
    
    # 2. Predictions
    for i, idx in enumerate(indices):
        ax = plt.subplot(2, n_samples, i + n_samples + 1)
        plt.imshow(lfw_images[idx].reshape(h, w), cmap='gray')
        pred = 'Arnold' if y_pred[idx] else 'Other'
        color = 'green' if y_pred[idx] == y_test[idx] else 'red'
        plt.title(f"Pred: {pred}", fontsize=10, color=color)
        plt.axis('off')
    
    plt.tight_layout()
    plt.suptitle(f'Arnold vs Others | Accuracy: {metrics["accuracy"]:.3f}', fontsize=16, y=0.98)
    plt.show()
    
    # 3. Eigenfaces in separate figure
    n_eigenfaces = min(12, pca.n_components_)
    fig2 = plt.figure(figsize=(16, 8))
    
    for i in range(n_eigenfaces):
        ax = plt.subplot(2, 6, i + 1)
        eigenface = pca.components_[i].reshape(h, w)
        plt.imshow(eigenface, cmap='gray')
        plt.title(f'Eigenface {i+1}', fontsize=9)
        plt.axis('off')
    
    plt.suptitle('First 12 Eigenfaces (PCA Components)', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.show()

def plot_arnold_samples(arnold_indices, y_pred_arnold, lfw_images, h, w):
    """Plot actual Arnold test images"""
    if len(arnold_indices) > 0:
        n_show = min(6, len(arnold_indices))
        fig, axes = plt.subplots(1, n_show, figsize=(5*n_show, 5))
        if n_show == 1: axes = [axes]
        
        for i, idx in enumerate(arnold_indices[:n_show]):
            axes[i].imshow(lfw_images[idx].reshape(h, w), cmap='gray')
            status = "✓ Correct" if y_pred_arnold[i] else "✗ Wrong"
            color = 'green' if y_pred_arnold[i] else 'red'
            axes[i].set_title(f'Arnold\n{status}', fontsize=11, color=color)
            axes[i].axis('off')
        
        plt.suptitle('Actual Arnold Test Images', fontsize=14)
        plt.tight_layout()
        plt.show()

