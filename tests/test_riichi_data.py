from __future__ import annotations
import unittest
import torch
from torch.utils.data import Dataset
import sys, pathlib
sys.path.append(str(pathlib.Path(__file__).resolve().parents[1] / "src"))
from visual.riichi_data import ChunkedRandomSampler


class _DummyDataset(Dataset):
    def __init__(self, length: int) -> None:
        self.length = length

    def __len__(self) -> int:
        return self.length

    def __getitem__(self, index: int):  # pragma: no cover - unused in sampler tests
        return index

class TestUtils(unittest.TestCase):
    def test_chunked_sampler_covers_all_indices(self) -> None:
        dataset = _DummyDataset(103)
        generator = torch.Generator().manual_seed(1234)
        sampler = ChunkedRandomSampler(dataset, chunk_size=7, generator=generator)

        order = list(iter(sampler))

        assert len(order) == 103
        assert len(set(order)) == 103
        assert sorted(order) == list(range(103))


    def test_chunked_sampler_is_seed_deterministic(self) -> None:
        dataset = _DummyDataset(50)
        generator_a = torch.Generator().manual_seed(42)
        generator_b = torch.Generator().manual_seed(42)

        order_a = list(iter(ChunkedRandomSampler(dataset, chunk_size=8, generator=generator_a)))
        order_b = list(iter(ChunkedRandomSampler(dataset, chunk_size=8, generator=generator_b)))

        assert order_a == order_b


    def test_chunked_sampler_handles_empty_dataset(self) -> None:
        dataset = _DummyDataset(0)
        sampler = ChunkedRandomSampler(dataset)

        assert list(iter(sampler)) == []
        assert len(sampler) == 0

if __name__ == "__main__":
    unittest.main()
