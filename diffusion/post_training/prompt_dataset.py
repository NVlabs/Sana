import json
import os

import torch
from torch.utils.data import Dataset, Sampler

_LOCAL_DATASET_ROOT = os.path.join(os.path.dirname(__file__), "dataset")


def _candidate_dataset_dirs(dataset):
    yield dataset

    dataset_name = os.path.basename(os.path.normpath(dataset))
    if not dataset_name:
        return

    local_dataset_dir = os.path.join(_LOCAL_DATASET_ROOT, dataset_name)
    if os.path.normpath(local_dataset_dir) != os.path.normpath(dataset):
        yield local_dataset_dir


def _resolve_dataset_file(dataset, filename):
    checked_paths = []
    for dataset_dir in _candidate_dataset_dirs(dataset):
        file_path = os.path.join(dataset_dir, filename)
        checked_paths.append(file_path)
        if os.path.exists(file_path):
            return file_path

    checked_str = ", ".join(checked_paths)
    raise FileNotFoundError(f"Could not find dataset file {filename!r}. Checked: {checked_str}")


class TextPromptDataset(Dataset):
    def __init__(self, dataset, split="train"):
        self.file_path = _resolve_dataset_file(dataset, f"{split}.txt")
        with open(self.file_path, encoding="utf-8") as f:
            self.prompts = [line.strip() for line in f.readlines()]

    def __len__(self):
        return len(self.prompts)

    def __getitem__(self, idx):
        return {"prompt": self.prompts[idx], "metadata": {}}

    @staticmethod
    def collate_fn(examples):
        prompts = [example["prompt"] for example in examples]
        metadatas = [example["metadata"] for example in examples]
        return prompts, metadatas


class GenevalPromptDataset(Dataset):
    def __init__(self, dataset, split="train"):
        self.file_path = _resolve_dataset_file(dataset, f"{split}_metadata.jsonl")
        with open(self.file_path, encoding="utf-8") as f:
            self.metadatas = [json.loads(line) for line in f]
            self.prompts = [item["prompt"] for item in self.metadatas]

    def __len__(self):
        return len(self.prompts)

    def __getitem__(self, idx):
        return {"prompt": self.prompts[idx], "metadata": self.metadatas[idx]}

    @staticmethod
    def collate_fn(examples):
        prompts = [example["prompt"] for example in examples]
        metadatas = [example["metadata"] for example in examples]
        return prompts, metadatas


class DistributedKRepeatSampler(Sampler):
    def __init__(self, dataset, batch_size, k, num_replicas, rank, seed=0):
        self.dataset = dataset
        self.batch_size = batch_size
        self.k = k
        self.num_replicas = num_replicas
        self.rank = rank
        self.seed = seed

        self.total_samples = self.num_replicas * self.batch_size
        assert (
            self.total_samples % self.k == 0
        ), f" total {self.total_samples} {self.k} k can not div n*b, k{k}-num_replicas{num_replicas}-batch_size{batch_size}"
        self.m = self.total_samples // self.k
        self.epoch = 0

    def __iter__(self):
        while True:
            g = torch.Generator()
            g.manual_seed(self.seed + self.epoch)
            indices = torch.randperm(len(self.dataset), generator=g)[: self.m].tolist()
            repeated_indices = [idx for idx in indices for _ in range(self.k)]

            shuffled_indices = torch.randperm(len(repeated_indices), generator=g).tolist()
            shuffled_samples = [repeated_indices[i] for i in shuffled_indices]

            per_card_samples = []
            for i in range(self.num_replicas):
                start = i * self.batch_size
                end = start + self.batch_size
                per_card_samples.append(shuffled_samples[start:end])
            yield per_card_samples[self.rank]

    def set_epoch(self, epoch):
        self.epoch = epoch
