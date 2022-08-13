import numpy as np
from torch.utils.data import Dataset, DataLoader

class MyDataset(Dataset):
    def __init__(self, name, numpy_samples, numpy_targets, transform=None):
        self.data = numpy_samples
        self.targets = numpy_targets
        self.name = name
        if transform:
            self.transform = transform
        else:
            self.transform = None
        self.dimensions = tuple(self.data.shape)

    def __getitem__(self, index):
        x = self.data[index]
        y = self.targets[index]

        if self.transform:
            # Channel dimension cant be used
            x_flat = x.astype(np.uint8).squeeze()
            x = self.transform(x_flat)

        return x, y

    def __len__(self):
        return len(self.data)

    def getDataLoader(self, batch_size=64):
        return DataLoader(self, batch_size=batch_size, shuffle=True)