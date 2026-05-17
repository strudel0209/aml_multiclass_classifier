"""Tests for score._pick_load_dir — picks the highest-step checkpoint."""
from score import _pick_load_dir


def test_picks_highest_step_checkpoint(tmp_path):
    for step in (10, 50, 30, 20):
        cp = tmp_path / f"checkpoint-{step}"
        cp.mkdir()
        (cp / "config.json").write_text("{}")
    assert _pick_load_dir(tmp_path).name == "checkpoint-50"


def test_root_with_config_short_circuits(tmp_path):
    """If model_root itself has config.json (e.g. mlflow.transformers final
    artefact dir), do *not* descend into checkpoint-*/."""
    (tmp_path / "config.json").write_text("{}")
    (tmp_path / "checkpoint-99").mkdir()
    (tmp_path / "checkpoint-99" / "config.json").write_text("{}")
    assert _pick_load_dir(tmp_path) == tmp_path


def test_no_checkpoints_returns_root(tmp_path):
    assert _pick_load_dir(tmp_path) == tmp_path


def test_malformed_checkpoint_name_does_not_crash(tmp_path):
    """A spurious 'checkpoint-foo' dir should be silently demoted (step=-1)
    rather than crashing the int() parse."""
    (tmp_path / "checkpoint-foo").mkdir()
    (tmp_path / "checkpoint-10").mkdir()
    (tmp_path / "checkpoint-10" / "config.json").write_text("{}")
    assert _pick_load_dir(tmp_path).name == "checkpoint-10"
