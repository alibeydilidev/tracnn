import sys
import os
import datetime

output_dir = os.path.join("outputs", "embedding_viz")
os.makedirs(output_dir, exist_ok=True)
timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import torch
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from models.tracnn import TraCNN
from data.utils import get_dataloaders

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load validation data
_, val_loader = get_dataloaders()

# Initialize model
model = TraCNN(input_channels=3, trans_dim=64, fused_dim=128, num_classes=5).to(device)
model.eval()

# Visualize positional encodings
with torch.no_grad():
    pos_enc = model.transformer.positional_encoding.squeeze(0).cpu().numpy()  # [512, D]
    pos_enc_pca = PCA(n_components=2).fit_transform(pos_enc)

plt.figure(figsize=(10, 6))
plt.plot(pos_enc_pca[:, 0], pos_enc_pca[:, 1], marker='o')
plt.title("Learned Positional Encodings (PCA)")
plt.xlabel("PC 1")
plt.ylabel("PC 2")
plt.grid(True)
plt.tight_layout()
title_slug = "positional_encodings"
plt.savefig(os.path.join(output_dir, f"{title_slug}_{timestamp}.png"))
plt.close()


# Load trained weights
model.load_state_dict(torch.load("outputs/tracnn_model_best.pt"))

# Preview rollout attention heatmap on first batch only
with torch.no_grad():
    for batch in val_loader:
        seq = batch["sequence"].to(device)
        _ = model.transformer(seq)
        rollout = model.transformer.get_attention_rollout()
        if rollout is not None:
            import seaborn as sns
            plt.figure(figsize=(10, 8))
            sns.heatmap(rollout.cpu().numpy(), cmap="viridis")
            plt.title("Transformer Attention Rollout Heatmap")
            plt.xlabel("From Token (Time Step)")
            plt.ylabel("To Token (Time Step)")
            plt.tight_layout()
            title_slug = "attention_rollout"
            plt.savefig(os.path.join(output_dir, f"{title_slug}_{timestamp}.png"))
            plt.close()
        break

# ------------------------------
# Layer-wise Attention Visualization
print("Visualizing layer-wise attention weights...")
with torch.no_grad():
    for batch in val_loader:
        seq = batch["sequence"].to(device)
        _ = model.transformer(seq)
        for i, layer in enumerate(model.transformer.layers):
            attn = layer.attn_weights
            if attn is not None:
                avg_attn = attn.mean(dim=(0, 1))  # Average over batch and heads
                plt.figure(figsize=(6, 5))
                import seaborn as sns
                sns.heatmap(avg_attn.cpu().numpy(), cmap="plasma")
                plt.title(f"Layer {i} Attention Heatmap")
                plt.xlabel("From Token")
                plt.ylabel("To Token")
                plt.tight_layout()
                title_slug = f"layer{i}_attention"
                plt.savefig(os.path.join(output_dir, f"{title_slug}_{timestamp}.png"))
                plt.close()
        break

# Store embeddings and labels
all_embeddings = []
all_labels = []

with torch.no_grad():
    for batch in val_loader:
        img = batch["image"].to(device)
        seq = batch["sequence"].to(device)
        labels = batch["label"].to(device)

        # Get embeddings from fusion layer
        cnn_out = model.cnn(img)
        trans_out = model.transformer(seq)
        fused = model.fusion(cnn_out, trans_out)

        all_embeddings.append(fused.cpu())
        all_labels.append(labels.cpu())

# Concatenate all batches
X = torch.cat(all_embeddings, dim=0).numpy()
y = torch.cat(all_labels, dim=0).numpy()

# Reduce to 2D with PCA
pca = PCA(n_components=2)
X_reduced = pca.fit_transform(X)

# Plot
plt.figure(figsize=(8, 6))
scatter = plt.scatter(X_reduced[:, 0], X_reduced[:, 1], c=y, cmap='tab10', alpha=0.7)
plt.legend(*scatter.legend_elements(), title="Classes")
plt.title("TraCNN Fusion Embeddings (PCA)")
plt.xlabel("PC 1")
plt.ylabel("PC 2")
plt.grid(True)
plt.tight_layout()
title_slug = "fusion_embeddings"
plt.savefig(os.path.join(output_dir, f"{title_slug}_{timestamp}.png"))
plt.close()

# ------------------------------
# CNN-only embedding visualization
cnn_embeddings = []
with torch.no_grad():
    for batch in val_loader:
        img = batch["image"].to(device)
        cnn_out = model.cnn(img)
        cnn_embeddings.append(cnn_out.cpu())

X_cnn = torch.cat(cnn_embeddings, dim=0).numpy()
X_cnn_pca = PCA(n_components=2).fit_transform(X_cnn)

plt.figure(figsize=(8, 6))
scatter = plt.scatter(X_cnn_pca[:, 0], X_cnn_pca[:, 1], c=y, cmap='tab10', alpha=0.7)
plt.legend(*scatter.legend_elements(), title="Classes")
plt.title("CNN Embeddings (PCA)")
plt.xlabel("PC 1")
plt.ylabel("PC 2")
plt.grid(True)
plt.tight_layout()
title_slug = "cnn_embeddings"
plt.savefig(os.path.join(output_dir, f"{title_slug}_{timestamp}.png"))
plt.close()

# ------------------------------
# CNN Saliency Visualization
print("Generating CNN saliency map...")
model.eval()
for batch in val_loader:
    img = batch["image"].to(device)
    label = batch["label"].to(device)

    img.requires_grad_()

    cnn_out = model.cnn(img)
    score = cnn_out[:, 0].mean()  # arbitrary: mean of first class embedding
    score.backward()

    saliency = img.grad.data.abs().mean(dim=1).cpu()  # [B, H, W]
    break  # only visualize first batch

plt.figure(figsize=(10, 4))
for i in range(min(4, saliency.shape[0])):
    plt.subplot(1, 4, i+1)
    plt.imshow(saliency[i], cmap='hot')
    plt.axis('off')
    plt.title(f"Sample {i}")
plt.suptitle("CNN Input Saliency (d|output|/dinput)")
plt.tight_layout()
title_slug = "cnn_saliency"
plt.savefig(os.path.join(output_dir, f"{title_slug}_{timestamp}.png"))
plt.close()

# ------------------------------
# Transformer-only embedding visualization
trans_embeddings = []
with torch.no_grad():
    for batch in val_loader:
        seq = batch["sequence"].to(device)
        trans_out = model.transformer(seq)
        trans_mean = trans_out.mean(dim=1)  # Aggregate sequence
        trans_embeddings.append(trans_mean.cpu())

X_trans = torch.cat(trans_embeddings, dim=0).numpy()
X_trans_pca = PCA(n_components=2).fit_transform(X_trans)

plt.figure(figsize=(8, 6))
scatter = plt.scatter(X_trans_pca[:, 0], X_trans_pca[:, 1], c=y, cmap='tab10', alpha=0.7)
plt.legend(*scatter.legend_elements(), title="Classes")
plt.title("Transformer Embeddings (PCA)")
plt.xlabel("PC 1")
plt.ylabel("PC 2")
plt.grid(True)
plt.tight_layout()
title_slug = "transformer_embeddings"
plt.savefig(os.path.join(output_dir, f"{title_slug}_{timestamp}.png"))
plt.close()
