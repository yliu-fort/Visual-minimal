from __future__ import annotations
import logging
from pathlib import Path
from typing import Optional
from datetime import datetime

from torch.utils.tensorboard import SummaryWriter

try:
    import mlflow
except Exception:  # pragma: no cover
    mlflow = None  # type: ignore


class RunLogger:
    """Unified management of logging、TensorBoard、(Optional) MLflow。"""

    def __init__(self, base_dir: str, exp_name: str, use_tb: bool, mlflow_cfg: Optional[dict] = None):
        ts = datetime.now().strftime("%Y%m%d-%H%M%S")
        self.run_dir = Path(base_dir) / f"{exp_name}-{ts}"
        self.run_dir.mkdir(parents=True, exist_ok=True)

        # Python logging
        self.logger = logging.getLogger("diffusion")
        self.logger.setLevel(logging.INFO)
        self.logger.handlers.clear()

        fh = logging.FileHandler(self.run_dir / "train.log")
        ch = logging.StreamHandler()
        fmt = logging.Formatter("%(asctime)s | %(levelname)s | %(message)s")
        fh.setFormatter(fmt)
        ch.setFormatter(fmt)
        self.logger.addHandler(fh)
        self.logger.addHandler(ch)

        # TensorBoard
        self.tb: Optional[SummaryWriter] = SummaryWriter(self.run_dir.as_posix()) if use_tb else None

        # MLflow (Optional)
        self.mlflow_on = bool(mlflow_cfg and mlflow_cfg.get("enabled"))
        if self.mlflow_on and mlflow is not None:
            mlflow.set_tracking_uri(mlflow_cfg["tracking_uri"])  # type: ignore[index]
            mlflow.set_experiment(mlflow_cfg["experiment_name"])  # type: ignore[index]
            self.mlr = mlflow.start_run(run_name=exp_name)  # type: ignore[assignment]
        else:
            self.mlr = None

    def log_metric(self, key: str, value: float, step: int) -> None:
        if self.tb:
            self.tb.add_scalar(key, value, step)
        if self.mlflow_on and self.mlr is not None and mlflow is not None:
            mlflow.log_metric(key, value, step=step)

    def log_params(self, params: dict) -> None:
        if self.tb:
            self.tb.add_text("params.yaml", str(params))
        if self.mlflow_on and self.mlr is not None and mlflow is not None:
            mlflow.log_params(params)

    def add_figure(self, tag: str, fig, step: int) -> None:  # type: ignore[no-untyped-def]
        if self.tb:
            self.tb.add_figure(tag, fig, step)

    def close(self) -> None:
        if self.tb:
            self.tb.flush()
            self.tb.close()
        if self.mlr is not None and mlflow is not None:  # pragma: no cover
            mlflow.end_run()
