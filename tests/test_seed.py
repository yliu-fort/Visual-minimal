import unittest
import sys, pathlib
sys.path.append(str(pathlib.Path(__file__).resolve().parents[1] / "src"))

import torch
from visual.seed import set_all_seeds

class TestUtils(unittest.TestCase):
    def test_deterministic_torch_randn(self):
        set_all_seeds(123, cudnn_benchmark=False, cudnn_deterministic=True)
        a = torch.randn(3, 3)
        set_all_seeds(123, cudnn_benchmark=False, cudnn_deterministic=True)
        b = torch.randn(3, 3)
        assert torch.allclose(a, b)

if __name__ == "__main__":
    unittest.main()