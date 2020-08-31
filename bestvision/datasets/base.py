import numpy as np
from torch.utils.data import Dataset


class ADataset(Dataset):
    def __matmul__(self, transform):
        assert callable(transform), "transform should be callable"
        class _DerivedClass(self.__class__):
            def __getitem__(self, index):
                item = super().__getitem__(index)
                item = transform(item)
                return item
        self.__class__ = _DerivedClass
        return self
