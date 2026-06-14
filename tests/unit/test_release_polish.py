from __future__ import annotations

import json
import runpy
import sys
import warnings
from pathlib import Path

import pytest

from optimizer.cli import main as cli_main
from optimizer.cli.main import load_obj, load_params, main
from optimizer.core.parameter import Parameter
from optimizer.core.parameter_space import ParameterSpace
from optimizer.distribution import build_zip, manifest
from optimizer.errors import ParameterValidationError


def test_distribution_manifest_reports_excluded_artifacts(tmp_path: Path) -> None:
    (tmp_path / "optimizer").mkdir()
    (tmp_path / "optimizer" / "version.py").write_text('__version__ = "4.0.0"\n')
    (tmp_path / "optimizer" / "__init__.py").write_text("")
    (tmp_path / ".pytest_cache").mkdir()
    (tmp_path / ".pytest_cache" / "state").write_text("cache")
    (tmp_path / "optimizer_results").mkdir()
    (tmp_path / "optimizer_results" / "trial.json").write_text("{}")
    (tmp_path / "dist").mkdir()
    (tmp_path / "dist" / "artifact.whl").write_text("wheel")
    (tmp_path / "old.zip").write_text("zip")

    report = manifest(tmp_path)

    assert report.hygiene_ok is False
    assert ".pytest_cache" not in report.forbidden_files
    assert ".pytest_cache/state" not in report.forbidden_files
    assert "optimizer_results" in report.forbidden_files
    assert "optimizer_results/trial.json" in report.forbidden_files
    assert "dist" in report.forbidden_files
    assert "dist/artifact.whl" in report.forbidden_files
    assert "old.zip" in report.forbidden_files


def test_distribution_zip_skips_release_artifacts(tmp_path: Path) -> None:
    (tmp_path / "optimizer").mkdir()
    (tmp_path / "optimizer" / "version.py").write_text('__version__ = "4.0.0"\n')
    (tmp_path / "optimizer" / "__init__.py").write_text("x = 1\n")
    (tmp_path / "build").mkdir()
    (tmp_path / "build" / "tmp.py").write_text("bad")
    output = tmp_path / "out.zip"

    built = build_zip(tmp_path, output)

    assert built == output
    assert manifest(tmp_path).hygiene_ok is False


def test_cli_load_params_validation_errors(tmp_path: Path) -> None:
    bad_json = tmp_path / "bad.json"
    bad_json.write_text("{")
    with pytest.raises(ValueError, match="invalid JSON"):
        load_params(bad_json)

    not_list = tmp_path / "not_list.json"
    not_list.write_text(json.dumps({"parameters": {"name": "x"}}))
    with pytest.raises(ValueError, match="parameter file must contain"):
        load_params(not_list)

    bad_row = tmp_path / "bad_row.json"
    bad_row.write_text(json.dumps(["not-object"]))
    with pytest.raises(ValueError, match="parameter row 0"):
        load_params(bad_row)


def test_cli_load_runner_validation_errors(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="FILE:OBJECT"):
        load_obj("runner.py")
    with pytest.raises(ValueError, match="FILE:OBJECT"):
        load_obj(str(tmp_path / "runner.py") + ":")

    runner_file = tmp_path / "runner.py"
    runner_file.write_text("value = 1\n")
    with pytest.raises(ValueError, match="cannot load runner module"):
        load_obj("runner_without_suffix:runner")
    with pytest.raises(ValueError, match="not callable"):
        load_obj(f"{runner_file}:value")


def test_cli_errors_are_clean_user_messages(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    params = tmp_path / "params.json"
    params.write_text("{")

    assert (
        main(["dry-run", "--params", str(params), "--output-dir", str(tmp_path)]) == 2
    )
    assert "invalid JSON" in capsys.readouterr().err

    valid = tmp_path / "params_valid.json"
    valid.write_text(
        json.dumps(
            [
                {
                    "name": "x",
                    "param_type": "int",
                    "default": 1,
                    "min_val": 1,
                    "max_val": 1,
                    "step": 1,
                }
            ]
        )
    )
    assert main(["run", "--params", str(valid), "--runner", "missing-format"]) == 2
    assert "FILE:OBJECT" in capsys.readouterr().err


def test_cli_main_module_execution_help(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(sys, "argv", ["optimizer.cli.main", "--help"])
    with pytest.raises(SystemExit) as exc, warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        runpy.run_module("optimizer.cli.main", run_name="__main__")
    assert exc.value.code == 0


def test_cli_package_main_delegates() -> None:
    with pytest.raises(SystemExit) as exc:
        cli_main.main(["--help"])
    assert exc.value.code == 0


def test_parameter_space_strict_parameter_validation() -> None:
    with pytest.raises(ParameterValidationError, match="unsupported parameter type"):
        ParameterSpace([Parameter("x", "integer", 1, 1, 2, 1)])  # type: ignore[arg-type]
    with pytest.raises(ParameterValidationError, match="must be numeric"):
        ParameterSpace([Parameter("x", "float", "bad", 1, 2, 1)])
    with pytest.raises(ParameterValidationError, match="must be numeric"):
        ParameterSpace([Parameter("x", "float", True, 1, 2, 1)])
    with pytest.raises(ParameterValidationError, match="max_val must be >= min_val"):
        ParameterSpace([Parameter("x", "float", 1.0, 2.0, 1.0, 1.0)])
    with pytest.raises(ParameterValidationError, match="integral"):
        ParameterSpace([Parameter("x", "int", 1.5, 1, 2, 1)])
    with pytest.raises(ParameterValidationError, match="bool default"):
        ParameterSpace([Parameter("flag", "bool", 1)])

    diagnostics = ParameterSpace([Parameter("x", "int", 1, 1, 2, 1)]).validate_params(
        {"x": True}
    )
    assert diagnostics[0].code == "PARAM_TYPE"
    float_diagnostics = ParameterSpace(
        [Parameter("x", "float", 1.0, 1.0, 2.0, 1.0)]
    ).validate_params({"x": False})
    assert float_diagnostics[0].code == "PARAM_TYPE"
