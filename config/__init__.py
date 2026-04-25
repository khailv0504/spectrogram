import json
from pathlib import Path
from typing import Any


DEFAULT_CONFIG_PATH = Path(__file__).resolve().parent / "config.yaml"
_PROJECT_ROOT = Path(__file__).resolve().parents[1]


def _resolve_project_path(value: str | Path) -> str:
    path = Path(value)
    if path.is_absolute():
        return str(path)
    return str((_PROJECT_ROOT / path).resolve())


def load_config(config_path: str | Path | None = None) -> dict[str, Any]:
    path = Path(config_path) if config_path is not None else DEFAULT_CONFIG_PATH

    with path.open("r", encoding="utf-8") as f:
        raw = f.read()

    try:
        config: dict[str, Any] = json.loads(raw)
    except json.JSONDecodeError as exc:
        try:
            import yaml  # type: ignore
        except ModuleNotFoundError as module_err:
            raise ValueError(
                "Config file is not valid JSON. Install PyYAML to use full YAML syntax."
            ) from module_err
        config = yaml.safe_load(raw)
        if not isinstance(config, dict):
            raise ValueError(f"Config file must produce a mapping/dict: {path}") from exc

    paths = config.setdefault("paths", {})
    output_dir_value = paths.get("output_dir", "output")
    paths["output_dir"] = _resolve_project_path(output_dir_value)

    train_root_dir_value = paths.get("train_root_dir")
    if train_root_dir_value is None:
        paths["train_root_dir"] = None
    elif isinstance(train_root_dir_value, (str, Path)):
        paths["train_root_dir"] = _resolve_project_path(train_root_dir_value)
    else:
        raise ValueError("paths.train_root_dir must be a path string or null.")

    paths["train_log_path"] = _resolve_project_path(
        paths.get("train_log_path", str(Path(output_dir_value) / "train_log.csv"))
    )
    paths["model_output_path"] = _resolve_project_path(
        paths.get("model_output_path", str(Path(output_dir_value) / "TrainedModel.pt"))
    )

    Path(paths["output_dir"]).mkdir(parents=True, exist_ok=True)
    Path(paths["train_log_path"]).parent.mkdir(parents=True, exist_ok=True)
    Path(paths["model_output_path"]).parent.mkdir(parents=True, exist_ok=True)

    config.setdefault("data", {})
    config.setdefault("training", {})
    config.setdefault("model", {})
    config.setdefault("visualization", {})

    visualization_cfg = config["visualization"]
    visualization_cfg["save_dir"] = _resolve_project_path(
        visualization_cfg.get("save_dir", output_dir_value)
    )
    Path(visualization_cfg["save_dir"]).mkdir(parents=True, exist_ok=True)

    return config
