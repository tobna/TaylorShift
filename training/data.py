"""
Module to load the datasets, using torch and datadings.
"""

from torchvision.datasets import CIFAR10, CIFAR100, StanfordCars, OxfordIIITPet, Flowers102
from datadings.reader import MsgpackReader
from datadings.torch import CompressedToPIL, Compose, Dataset
import torchvision.transforms as tv_transforms
from torch.utils.data import DataLoader, DistributedSampler
from datasets.distilled_dataset import DistilledImagenette
from datasets.long_listops import LongListOps
from datasets.imdb import IMDB, ByteEncoder
from datasets.pathfinder import Pathfinder
from datasets.cityscapes import Cityscapes
from datasets.ADE20k import ADE20k
from datasets.data_utils import three_augment, segment_augment, collate_imnet, collate_listops, ToOneHotSequence
from datasets.animal_net import AnimalNet2k


def prepare_dataset(dataset_name, args, transform=None, train=True, rank=None):
    """Load a dataset from disk, different formats are used for different datasets.

    Supported datasets: CIFAR10, ImageNet, ImageNet21k

    Parameters
    ----------
    dataset_name : str
        name of the dataset
    args
        further arguments
    transform : list[Module] | str
        transformations to use on the data; the list gets composed, or give args.augment_strategy
    train : bool
        use the training split (or test/validation split)
    rank : int
        global rank of this process in distributed training

    Returns
    -------
    tuple[DataLoader, int]
        the data loader, and number of classes
    """
    if transform is None:
        if args.augment_strategy == "3-augment":
            transform = three_augment(args, as_list=True, test=not train)
        elif args.augment_strategy == "differentiable-transform":
            from datasets.distilled_dataset import differentiable_augment

            transform = differentiable_augment(args, as_list=True, test=not train)
        elif args.augment_strategy == "none":
            transform = []
        elif args.augment_strategy == "lm_one_hot":
            transform = [tv_transforms.Grayscale(num_output_channels=1), tv_transforms.ToTensor(), ToOneHotSequence()]
        elif args.augment_strategy == "segment-augment":
            transform = segment_augment(args, test=not train)
        else:
            raise NotImplementedError(f"Augmentation strategy {args.augment_strategy} is not implemented (yet).")

    dataset_name = dataset_name.lower()
    ignore_index = -100
    if dataset_name == "cifar10":
        dataset = CIFAR10(
            root=args.dataset_root + "CIFAR", train=train, download=False, transform=tv_transforms.Compose(transform)
        )
        n_classes, collate = 10, None

    elif dataset_name == "cifar100":
        dataset = CIFAR100(
            root=args.dataset_root + "CIFAR", train=train, download=False, transform=tv_transforms.Compose(transform)
        )
        n_classes, collate = 100, None

    elif dataset_name == "stanford-cars":
        dataset = StanfordCars(
            root=args.dataset_root,
            split="train" if train else "test",
            download=False,
            transform=tv_transforms.Compose(transform),
        )
        n_classes, collate = 196, None

    elif dataset_name == "oxford-pet":
        dataset = OxfordIIITPet(
            root=args.dataset_root,
            split="trainval" if train else "test",
            download=False,
            transform=tv_transforms.Compose(transform),
        )
        n_classes, collate = 37, None

    elif dataset_name == "flowers102":
        dataset = Flowers102(
            root=args.dataset_root,
            split="train" if train else "test",
            download=False,
            transform=tv_transforms.Compose(transform),
        )
        n_classes, collate = 102, None

    elif dataset_name == "places365":
        reader = MsgpackReader(
            f"{args.dataset_root}Places365/large_challenge/{'training' if train else 'validation'}.msgpack"
        )
        dataset = Dataset(reader, transforms={"image": Compose([CompressedToPIL()] + transform)})

        n_classes, collate = 365, collate_imnet

    elif dataset_name in ["imagenet", "imagenet21k"]:
        reader = MsgpackReader(f"{args.dataset_root}imagenet/msgpack/{'train' if train else 'val'}.msgpack")
        if "21k" in dataset_name:
            reader = MsgpackReader(f"{args.dataset_root}imagenet21k/{'train' if train else 'val'}.msgpack")

        dataset = Dataset(reader, transforms={"image": Compose([CompressedToPIL()] + transform)})

        n_classes, collate = 1000, collate_imnet
        if "21k" in dataset_name:
            n_classes = 10_450

    elif dataset_name.startswith("distilledimagenette"):
        ipc_folder = dataset_name.replace("_", "-").split("-")[1]
        dataset = DistilledImagenette(
            distill_dataset=f"/netscratch/raue/transformers_distilled_dataset/imagenette/{ipc_folder}",
            train=train,
            transform=Compose(transform),
        )
        n_classes, collate = 10, None

    elif dataset_name == "imagenette":
        reader = MsgpackReader(f"{args.dataset_root}imagenette/{'train' if train else 'val'}.msgpack")
        dataset = Dataset(reader, transforms={"image": Compose([CompressedToPIL()] + transform)})

        n_classes, collate = 10, collate_imnet

    elif dataset_name.startswith("listops"):
        dataset_params = dataset_name.split("_")
        if len(dataset_params) == 2:
            min_len = max_len = int(dataset_params[1])
        else:
            min_len, max_len = int(dataset_params[1]), int(dataset_params[2])
        dataset = LongListOps(
            epoch_length=96_000 if train else 50_000,
            min_length=min_len,
            max_length=max_len,
            batch_size=args.batch_size,
            batch_mode=True,
        )

        n_classes, collate = 10, collate_listops

    elif dataset_name == "imdb_byte":
        dataset = IMDB(
            args.dataset_root + "IMDB/aclImdb_v1.tar",
            train=train,
            transform=tv_transforms.Compose([ByteEncoder(pad_to_length=4_000, train=train)] + transform),
        )
        n_classes, collate = 2, None

    elif dataset_name == "cifar10_pixel":
        # 32 x 32 px
        # 256 8-bit grayscale values
        dataset = CIFAR10(
            root=args.dataset_root + "CIFAR", train=train, download=False, transform=tv_transforms.Compose(transform)
        )
        n_classes, collate = 10, None

    elif dataset_name == "pathfinder":
        dataset = Pathfinder(
            args.dataset_root + "pathfinder/pathfinder32", train=train, transform=tv_transforms.Compose(transform)
        )
        n_classes, collate = 2, None

    elif dataset_name == "pathfinder-x":
        dataset = Pathfinder(
            args.dataset_root + "pathfinder/pathfinder128", train=train, transform=tv_transforms.Compose(transform)
        )
        n_classes, collate = 2, None

    elif dataset_name == "cityscapes":
        dataset = Cityscapes(
            args.dataset_root + "Cityscapes",
            split="train" if train else "val",
            mode="fine",
            transform=tv_transforms.Compose(transform),
            imsize=args.imsize,
        )
        n_classes, collate = 20, None
        ignore_index = 0

    elif dataset_name == "ade20k":
        dataset = ADE20k(
            args.dataset_root + "ADE20k1",
            split="training" if train else "validation",
            transform=tv_transforms.Compose(transform),
        )
        n_classes, collate = 151, None
        ignore_index = 0

    elif dataset_name.startswith("animalnet"):
        error_rate = 0.0
        if "_e" in dataset_name:
            error_rate = int(dataset_name.split("_e")[1][:2]) / 100.0
        dataset = AnimalNet2k(
            args.dataset_root + "AnimalNet2k",
            train=train,
            transform=tv_transforms.Compose(transform),
            shift_up_prob=error_rate,
        )
        n_classes, collate = dataset.n_classes, None

    else:
        raise NotImplementedError(f"Dataset {dataset_name} is not implemented (yet).")

    if args.distributed:
        sampler = DistributedSampler(dataset, num_replicas=args.world_size, rank=rank, shuffle=train and args.shuffle)
    else:
        sampler = None

    loader_batch_size = 1 if dataset_name.startswith("listops") else args.batch_size

    data_loader = DataLoader(
        dataset,
        batch_size=loader_batch_size,
        pin_memory=args.pin_memory,
        num_workers=args.num_workers,
        drop_last=train,
        prefetch_factor=args.prefetch_factor if args.num_workers > 0 else 2,
        persistent_workers=(args.num_workers > 0),
        collate_fn=collate,
        shuffle=None if sampler else train and args.shuffle,
        sampler=sampler,
    )
    return data_loader, n_classes, ignore_index
