import torch
from torch.utils.data import Dataset
from random import choice, randint


class MathTree:
    MIN = "[MIN"
    MAX = "[MAX"
    MED = "[MED"
    FIRST = "[FIRST"
    LAST = "[LAST"
    SUM_MOD = "[SM"
    END = "]"

    OPERATORS = [MIN, MAX, MED, FIRST, LAST, SUM_MOD]
    VALUES = list(range(10))
    VALUE_P = 0.25

    def __init__(self, op=None):
        self.op = choice(self.OPERATORS) if op is None else op
        assert self.op in self.OPERATORS, f"Invalid operator {self.op}"
        self.data = []

    def __len__(self):
        # 2 tokens for the operator and the closing bracket
        return sum(len(d) if isinstance(d, MathTree) else 1 for d in self.data) + 2

    def value(self):
        if self.op == self.MIN:
            return min(d.value() if isinstance(d, MathTree) else d for d in self.data)
        if self.op == self.MAX:
            return max(d.value() if isinstance(d, MathTree) else d for d in self.data)
        if self.op == self.MED:
            return sorted(
                d.value() if isinstance(d, MathTree) else d for d in self.data
            )[len(self.data) // 2]
        if self.op == self.FIRST:
            return (
                self.data[0].value()
                if isinstance(self.data[0], MathTree)
                else self.data[0]
            )
        if self.op == self.LAST:
            return (
                self.data[-1].value()
                if isinstance(self.data[-1], MathTree)
                else self.data[-1]
            )
        if self.op == self.SUM_MOD:
            return (
                sum(d.value() if isinstance(d, MathTree) else d for d in self.data) % 10
            )

    def sequence(self):
        return (
            [self.op]
            + sum(
                [d.sequence() if isinstance(d, MathTree) else [d] for d in self.data],
                [],
            )
            + [self.END]
        )

    def __str__(self):
        return " ".join(str(d) for d in self.sequence())

    def _numerical_entries(self, depth=1, max_depth=10):
        if depth < max_depth:
            return sum(
                (
                    d._numerical_entries(depth=depth + 1, max_depth=max_depth)
                    if isinstance(d, MathTree)
                    else 1
                )
                for d in self.data
            )
        return sum(1 for d in self.data if d in self.VALUES)

    def _init_tree_vals(self, min_sub_len, max_sub_len, max_args):
        list_len = randint(min_sub_len - 2, min(max_sub_len - 2, max_args))
        self.data = [choice(self.VALUES) for _ in range(list_len)]

    def _add_subtree(self, max_sub_len, max_args, min_sub_len=4, depth=1, max_depth=10):
        assert min_sub_len >= 4, "Minimum subtree length must be at least 4"
        if len(self.data) == 0:
            self._init_tree_vals(min_sub_len, max_sub_len, max_args)
            return

        # get equally distributed index, but don't make the tree too deep
        num_entries = [
            (
                d._numerical_entries(depth=depth + 1, max_depth=max_depth)
                if isinstance(d, MathTree)
                else 1
            )
            for d in self.data
        ]
        assert (
            sum(num_entries) > 0
        ), f"No numerical entries found in {self} @ depth {depth}"
        idx_weight = randint(1, sum(num_entries))
        idx = -1
        for i, n in enumerate(num_entries):
            if idx_weight <= n:
                idx = i
                break
            idx_weight -= n
        assert idx >= 0, "No index found"

        if isinstance(self.data[idx], MathTree):
            self.data[idx]._add_subtree(
                max_sub_len, max_args, min_sub_len, depth + 1, max_depth
            )
            return

        # add a new subtree where there was a number before
        subtree = MathTree()
        subtree._init_tree_vals(min_sub_len, max_sub_len, max_args)
        self.data[idx] = subtree

    def _open_slots(self, max_args, depth=1, max_depth=10):
        slots = max_args - len(self.data)
        if depth < max_depth:
            slots += sum(
                d._open_slots(max_args, depth=depth + 1)
                for d in self.data
                if isinstance(d, MathTree)
            )
        return slots

    def _add_slot(self, max_args, depth=1, max_depth=10):
        if self._open_slots(max_args, depth=depth + 1, max_depth=max_depth) == 0:
            return

        # get equally distributed index, but don't make the tree too deep
        open_slots = [max_args - len(self.data)]
        if depth < max_depth:
            open_slots += [
                (
                    d._open_slots(max_args, depth=depth + 1, max_depth=max_depth)
                    if isinstance(d, MathTree)
                    else 0
                )
                for d in self.data
            ]

        idx_weight = randint(1, sum(open_slots))
        idx = -1
        for i, n in enumerate(open_slots):
            if idx_weight <= n:
                idx = i
                break
            idx_weight -= n
        assert idx >= 0, "No index found"

        if idx == 0:
            self.data.append(choice(self.VALUES))
            return
        self.data[idx - 1]._add_slot(max_args, depth=depth + 1, max_depth=max_depth)

    @classmethod
    def generate(cls, length, max_depth=10, max_args=10):
        tree = cls()

        while len(tree) < length - 4:
            tree._add_subtree(
                length - len(tree), max_args=max_args, max_depth=max_depth
            )

        if (
            len(tree) < length
            and tree._open_slots(max_args=max_args, max_depth=max_depth) == 0
        ):
            return cls.generate(length, max_depth=max_depth, max_args=max_args)

        # need to add the last nodes to some subtrees
        while len(tree) < length:
            tree._add_slot(max_args=max_args, max_depth=max_depth)

        return tree

    @classmethod
    def encode(cls, item):
        encoding = [0 for _ in range(10 + len(cls.OPERATORS) + 1)]
        if isinstance(item, int):
            assert 0 <= item <= 9, f"Invalid value {item}"
            encoding[item] = 1
        elif item in cls.OPERATORS:
            encoding[10 + cls.OPERATORS.index(item)] = 1
        else:
            assert item == cls.END, f"Invalid item {item}"
            encoding[-1] = 1
        return encoding

    def to_tensor(self):
        return torch.tensor(
            [self.encode(it) for it in self.sequence()], dtype=torch.float
        )


class LongListOps(Dataset):
    def __init__(
        self,
        epoch_length=1_600,
        batch_size=1,
        max_depth=10,
        max_args=10,
        max_length=2000,
        min_length=500,
        batch_mode=None,
    ):
        super().__init__()
        self.epoch_length = epoch_length
        self.max_depth = max_depth
        self.max_args = max_args
        self.max_length = max_length
        self.min_length = min_length
        self.batch_size = batch_size
        if batch_mode is None:
            batch_mode = batch_size > 1
        assert (
            batch_mode or batch_size == 1
        ), "Batch mode must be enabled to use batch_size > 1"
        self.batch_mode = batch_mode

    def __len__(self):
        return self.epoch_length // self.batch_size

    def __getitem__(self, item):
        if item >= len(self):
            raise IndexError(
                f"Index {item} out of bounds for dataset of length {len(self)} / {self.epoch_length}"
            )

        length = randint(self.min_length, self.max_length)

        if self.batch_mode:
            trees = [
                MathTree.generate(
                    length, max_depth=self.max_depth, max_args=self.max_args
                )
                for _ in range(self.batch_size)
            ]
            return torch.stack(
                [tree.to_tensor() for tree in trees], dim=0
            ), torch.stack(
                [torch.tensor(tree.value(), dtype=torch.long) for tree in trees], dim=0
            )

        tree = MathTree.generate(
            length, max_depth=self.max_depth, max_args=self.max_args
        )
        return tree.to_tensor(), torch.tensor(tree.value(), dtype=torch.long)
