import unittest
import sys, pathlib
sys.path.append(str(pathlib.Path(__file__).resolve().parents[1] / "src"))
from visual.config import load_config

class TestUtils(unittest.TestCase):
    def test_load_config_ok(self):
        cfg = load_config("configs/default.yaml")
        assert cfg.run.seed == 42
        assert cfg.data.batch_size > 0

if __name__ == "__main__":
    unittest.main()