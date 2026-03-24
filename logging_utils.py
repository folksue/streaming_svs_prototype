from __future__ import annotations

from typing import Any, Dict, Iterable, List

from utils import ensure_dir


class BaseMetricLogger:
    def log_config(self, cfg: Dict[str, Any]) -> None:
        return None

    def log_metrics(self, metrics: Dict[str, float], step: int) -> None:
        return None

    def close(self) -> None:
        return None


class ConsoleMetricLogger(BaseMetricLogger):
    pass


class TensorBoardMetricLogger(BaseMetricLogger):
    def __init__(self, log_dir: str):
        from torch.utils.tensorboard import SummaryWriter

        ensure_dir(log_dir)
        self.writer = SummaryWriter(log_dir=log_dir)

    def log_config(self, cfg: Dict[str, Any]) -> None:
        self.writer.add_text("config", repr(cfg))

    def log_metrics(self, metrics: Dict[str, float], step: int) -> None:
        for key, value in metrics.items():
            self.writer.add_scalar(key, value, step)

    def close(self) -> None:
        self.writer.close()


class WandbMetricLogger(BaseMetricLogger):
    def __init__(self, project: str, run_name: str | None, cfg: Dict[str, Any], mode: str):
        import wandb

        self.wandb = wandb
        self.wandb.init(project=project, name=run_name, config=cfg, mode=mode)

    def log_metrics(self, metrics: Dict[str, float], step: int) -> None:
        self.wandb.log(metrics, step=step)

    def close(self) -> None:
        self.wandb.finish()


class CompositeMetricLogger(BaseMetricLogger):
    def __init__(self, backends: Iterable[BaseMetricLogger]):
        self.backends = list(backends)

    def log_config(self, cfg: Dict[str, Any]) -> None:
        for backend in self.backends:
            backend.log_config(cfg)

    def log_metrics(self, metrics: Dict[str, float], step: int) -> None:
        for backend in self.backends:
            backend.log_metrics(metrics, step)

    def close(self) -> None:
        for backend in self.backends:
            backend.close()


def build_metric_logger(cfg: Dict[str, Any]) -> CompositeMetricLogger:
    log_cfg = cfg.get("logging", {})
    backends = parse_backends(log_cfg.get("backends", ["console"]))
    supported = {"console", "tensorboard", "wandb"}
    unknown = [x for x in backends if x not in supported]
    if unknown:
        raise ValueError(f"Unsupported logging backends: {unknown}. Supported backends: {sorted(supported)}")

    logger_backends: List[BaseMetricLogger] = []
    if "console" in backends:
        logger_backends.append(ConsoleMetricLogger())
    if "tensorboard" in backends:
        logger_backends.append(TensorBoardMetricLogger(log_dir=log_cfg.get("tensorboard_dir", "runs/default")))
    if "wandb" in backends:
        logger_backends.append(
            WandbMetricLogger(
                project=log_cfg.get("wandb_project", "streaming-svs"),
                run_name=log_cfg.get("run_name"),
                cfg=cfg,
                mode=log_cfg.get("wandb_mode", "online"),
            )
        )
    return CompositeMetricLogger(logger_backends)


def parse_backends(raw: Any) -> List[str]:
    if isinstance(raw, str):
        values = [raw]
    elif isinstance(raw, list):
        values = raw
    else:
        raise TypeError("logging.backends must be a string or list")
    normalized = [str(x).strip().lower() for x in values if str(x).strip()]
    if not normalized:
        return ["console"]
    return normalized
