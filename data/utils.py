import yaml

with open("training/config.yaml", "r") as f:
    config = yaml.safe_load(f)

def custom_collate_fn(batch):
    from torch.utils.data._utils.collate import default_collate
    elem = batch[0]
    output = {}
    for key in elem:
        if key == "edge_index":
            output[key] = [d[key] for d in batch]  # keep as list
        else:
            output[key] = default_collate([d[key] for d in batch])
    return output

from torch.utils.data import DataLoader
from data.dataset import TraCNNDataset

def get_dataloaders():
    train_dataset = TraCNNDataset(num_samples=config["train_samples"])
    val_dataset = TraCNNDataset(num_samples=config["val_samples"])

    train_loader = DataLoader(train_dataset, batch_size=config["batch_size"], shuffle=True, collate_fn=custom_collate_fn)
    val_loader = DataLoader(val_dataset, batch_size=config["batch_size"], shuffle=False, collate_fn=custom_collate_fn)

    return train_loader, val_loader

if __name__ == "__main__":
    train_loader, _ = get_dataloaders()
    for batch in train_loader:
        print(batch['image'].shape)     # [B, C, H, W]
        print(batch['sequence'].shape)  # [B, T, D]
        print(batch['label'].shape)     # [B]
        break