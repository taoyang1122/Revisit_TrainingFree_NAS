from torch.utils.data.dataset import Dataset


class repeatDataset(Dataset):
    # this only works when dataloader batchsize equal to the base dataset size
    def __init__(self, dataset, repeat=200):
        self.dataset = dataset
        self.repeat = repeat
        self.base_len = len(dataset)

    def __getitem__(self, idx):
        base_idx = idx % self.base_len
        return self.dataset[base_idx]

    def __len__(self):
        return self.base_len * self.repeat
