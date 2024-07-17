import random
import tarfile
import torch
from torch.utils.data import Dataset


class IMDB(Dataset):
    def __init__(self, data_file, train=True, transform=None, target_transform=None):
        self.file_name = data_file
        self.split = "train" if train else "test"
        self.data_file = tarfile.open(self.file_name, "r")
        pos = [
            f
            for f in self.data_file.getmembers()
            if f.name.startswith(f"aclImdb/{self.split}/pos")
            and f.name.endswith(".txt")
        ]
        neg = [
            f
            for f in self.data_file.getmembers()
            if f.name.startswith(f"aclImdb/{self.split}/neg")
            and f.name.endswith(".txt")
        ]
        self.file_list = []
        self.data = []
        # for f in tqdm(self.data_file.extractall(path=f'aclImdb/{self.split}'), total=50_000):
        for f_pos, f_neg in zip(pos, neg):
            # self.file_list.append((f_pos, 1))
            # self.file_list.append((f_neg, 0))
            # print(f)
            self.data.append(
                (
                    " ".join(
                        str(f).strip()
                        for f in self.data_file.extractfile(f_pos).readlines()
                    ),
                    1,
                )
            )
            self.data.append(
                (
                    " ".join(
                        str(f).strip()
                        for f in self.data_file.extractfile(f_neg).readlines()
                    ),
                    0,
                )
            )

        self.transform = transform
        self.target_transform = target_transform

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        # f_name, label = self.file_list[idx]
        # print(f"getting {idx} -> {f_name}")
        # text = ' '.join(str(f).strip() for f in self.data_file.extractfile(f_name).readlines())

        text, label = self.data[idx]

        if self.transform is not None:
            text = self.transform(text)

        if self.target_transform is not None:
            label = self.target_transform(label)

        return text, torch.tensor(label, dtype=torch.long)

    def __del__(self):
        try:
            self.data_file.close()
        except AttributeError:
            pass


class ByteEncoder:
    def __init__(self, pad_to_length=None, pad_value=0, to_tensor=True, train=False):
        self.pad_to_length = pad_to_length
        self.pad_value = pad_value
        self.to_tensor = to_tensor
        self.train = train

    def __call__(self, *args, **kwargs):
        return self.encode(*args, **kwargs)

    def encode(self, text):
        encoded = list(bytearray(text, "utf-8"))

        if self.pad_to_length:
            if len(encoded) > self.pad_to_length:
                if self.train:
                    start_idx = random.randint(0, len(encoded) - self.pad_to_length)
                    encoded = encoded[start_idx : start_idx + self.pad_to_length]
                else:
                    encoded = encoded[: self.pad_to_length]
            else:
                encoded += [self.pad_value] * max(0, self.pad_to_length - len(encoded))

        if self.to_tensor:
            enc_tensor = torch.zeros(len(encoded), 256)
            for i, val in enumerate(encoded):
                enc_tensor[i, val] = 1
            return enc_tensor

        return encoded
