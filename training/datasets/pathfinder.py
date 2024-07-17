# look into https://github.com/drewlinsley/pathfinder/tree/master
import io
import tarfile

import PIL
from PIL import Image
from torch.utils.data import Dataset, get_worker_info
import os
import numpy as np

# *.npy is array of (folder, file, id, label, ... (some metadata))


class Pathfinder(Dataset):
    def __init__(
        self, folder, train=True, transform=None, target_transform=None, **kwargs
    ):
        super().__init__(**kwargs)
        self.folder = folder
        self.train = train
        self.transform = transform
        self.target_transform = target_transform

        meta_files = sorted(os.listdir(os.path.join(self.folder, "metadata")))
        if train:
            meta_files = meta_files[: int(len(meta_files) * 0.8)]
            # print(f"Using files {meta_files} for training")
        else:
            meta_files = meta_files[int(len(meta_files) * 0.8) :]

        self.data = []
        for meta_file in meta_files:
            if meta_file.endswith(".npy"):
                np_list = np.load(os.path.join(self.folder, "metadata", meta_file))
                for np_arr in np_list:
                    self.data.append(
                        (os.path.join(np_arr[0], np_arr[1]), np_arr[2], np_arr[3])
                    )
                # print(f"load {meta_file} with {len(np_list)} entries")

        # open one file per worker
        worker = get_worker_info()
        worker = worker.id if worker is not None else None
        self.tar_handles = {
            worker: tarfile.open(os.path.join(self.folder, "imgs.tar"), "r")
        }

        # caching for getmember
        self.tar_handles[worker].getmembers()

    def __len__(self):
        return len(self.data)

    def __del__(self):
        for tar_handle in self.tar_handles.values():
            try:
                tar_handle.close()
            except AttributeError:
                pass

    def __getitem__(self, idx):
        path, id, label = self.data[idx]
        worker = get_worker_info()
        worker = worker.id if worker is not None else None
        if worker not in self.tar_handles:
            self.tar_handles[worker] = tarfile.open(
                os.path.join(self.folder, "imgs.tar"), "r"
            )
            self.tar_handles[worker].getmembers()

        try:
            image = Image.open(
                io.BytesIO(self.tar_handles[worker].extractfile(path).read())
            )
        except PIL.UnidentifiedImageError as e:
            print(
                f"UnidentifiedImageError loading image {path}, id {id}, label {label} from file {self.img_file}"
            )
            raise e
        except tarfile.ReadError as e:
            print(
                f"ReadError loading image {path}, id {id}, label {label} from file {self.img_file}"
            )
            raise e
        except Exception as e:
            print(
                f"Error loading image {path}, id {id}, label {label} from file {self.img_file}"
            )
            raise e

        label = int(label)
        if self.transform is not None:
            image = self.transform(image)
        if self.target_transform is not None:
            label = self.target_transform(label)
        return image, label
