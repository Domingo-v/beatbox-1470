import os
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

class NumpySpectrogramDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        """
        root_dir/
          genre1/
            file1.npy
            file2.npy
          genre2/
            ...
        """
        self.samples = []
        self.labels  = []
        self.genres  = sorted(
            d for d in os.listdir(root_dir)
            if os.path.isdir(os.path.join(root_dir, d))
        )
        self.genre_to_idx = {g:i for i,g in enumerate(self.genres)}
        for g in self.genres:
            gdir = os.path.join(root_dir, g)
            for fn in os.listdir(gdir):
                if fn.endswith('.npy'):
                    self.samples.append(os.path.join(gdir, fn))
                    self.labels.append(self.genre_to_idx[g])
        self.transform = transform

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        spec = np.load(self.samples[idx]) # shape (96, T)
        spec = torch.from_numpy(spec).float()# to float tensor
        spec = spec.unsqueeze(0) # add channel: (1,96,T)
        label = self.labels[idx]
        if self.transform:
            spec = self.transform(spec)
        return spec, label

class DummyCNN(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1,  8, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(8, 16, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((1,1)),  # collapse H×W→1×1
        )
        self.classifier = nn.Linear(16, num_classes)

    def forward(self, x):
        x = self.features(x)            # → (batch,16,1,1)
        x = x.view(x.size(0), -1)       # → (batch,16)
        return self.classifier(x)       # → (batch,num_classes)

def main():
    DATA_ROOT = "processed_data"      # point this at your npy folder
    BATCH_SIZE = 16

    dataset = NumpySpectrogramDataset(DATA_ROOT)
    loader  = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)

    print(f"Found genres: {dataset.genres}")
    print(f"Total samples: {len(dataset)}")

    model = DummyCNN(num_classes=len(dataset.genres))

    # grab one batch
    specs, labels = next(iter(loader))
    print(f"Input batch shape: {specs.shape}")     # (B,1,96,T)
    print(f"Labels shape:       {labels.shape}")    # (B,)

    # forward pass
    outputs = model(specs)
    print(f"Output shape:      {outputs.shape}")    # (B,num_genres)

if __name__ == "__main__":
    main()
