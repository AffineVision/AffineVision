from torch.utils.data import Dataset


class ADataset(Dataset):
    def __matmul__(self, transform):
        return TDataset(self, transform)


class TDataset(ADataset):
    def __init__(self, source, transform):
        self.source = source
        assert callable(transform), "transform should be callable!"
        self.transform = transform

    def __len__(self):
        return len(self.source)

    def __getitem__(self, item):
        item = self.source[item]
        return self.transform(item)
