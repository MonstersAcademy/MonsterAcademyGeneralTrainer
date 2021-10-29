from torch.utils.data import Dataset


class ImageDataset(Dataset):
    def __init__(self, setFolder, transforms) -> None:
        super().__init__()
        self.setFolder = setFolder
        self.transforms = transforms

    def __len__(self):
        pass

    def __getitem__(self, index):
        return super().__getitem__(index)
