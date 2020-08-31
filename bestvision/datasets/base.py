import copy
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
        result = copy.copy(self)
        result.__class__ = _DerivedClass
        return result
       
class MyDataset(ADataset):
    def __init__(self, arg1):
        self.arg1 = arg1
        self.data = np.arange(100)

    def __len__(self):
        return self.data.size
    
    @property
    def name(self):
        return "This is MyDataset"
    
    def __getitem__(self, index):
        return self.data[index]

if __name__ == "__main__":
    
    def func(x):
        return x * 2
    
    def func3(x):
        return x ** 2

    data = MyDataset(2)

    data2 = (data @ func) @ func3
    data3 = data @ func3

    print(type(data2))
    print(type(data3))

    print(type(data2) == type(data3))

    for v1, v2 in zip(data2, data3):
        print(v1, v2)

