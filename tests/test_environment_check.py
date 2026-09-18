"""CPU-safe tests for the diagnostic itself, not a substitute for the GPU gate."""
import json
from pathlib import Path

import pytest
import torch
import torchvision

from training.check_environment import (
    PACKAGES, Report, basic_library_test, check_tensors, nms_test,
    require, tensor_and_conv_test, tensor_leaves, write_json,
)


def test_require_raises():
    with pytest.raises(RuntimeError, match="failure"):
        require(False, "failure")


def test_report_failure_is_recorded():
    report = Report()
    assert not report.run("broken", lambda: require(False, "expected"))
    assert report.failed
    assert "traceback" in report.checks[0]


def test_warning_does_not_equal_failure():
    report = Report()
    report.add("WARN", "free space", "low")
    assert not report.failed


def test_json_write_is_utf8_and_replaces_previous(tmp_path: Path):
    path = tmp_path / "nested" / "report.json"
    write_json(path, {"status": "PASS"})
    write_json(path, {"status": "RUNNING", "phase_1_1_passed": False})
    assert json.loads(path.read_text(encoding="utf-8"))["status"] == "RUNNING"
    assert not path.with_name(path.name + ".tmp").exists()


def test_nested_tensor_outputs():
    data = {"a": [torch.ones(1), (torch.zeros(2), None)]}
    assert len(list(tensor_leaves(data, torch))) == 2
    assert check_tensors(data, torch, torch.device("cpu")) == [[1], [2]]


@pytest.mark.parametrize("output", [None, torch.empty(0), torch.tensor([float("nan")]),
                                   torch.tensor([float("inf")])])
def test_bad_outputs_are_rejected(output):
    with pytest.raises(RuntimeError):
        check_tensors(output, torch, torch.device("cpu"))


def test_cnn_helper_on_cpu():
    torch.set_num_threads(2)
    result = tensor_and_conv_test(torch, torch.device("cpu"))
    assert result["output_shape"] == [1, 2]
    assert result["device"] == "cpu"


def test_nms_helper_on_cpu():
    assert nms_test(torch, torchvision, torch.device("cpu"))["kept_indices"] == [0, 2]


def test_library_functions():
    import cv2
    import numpy
    import pandas
    import yaml
    result = basic_library_test({"torch": torch, "cv2": cv2, "numpy": numpy,
                                 "pandas": pandas, "yaml": yaml})
    assert all(result.values())


def test_cuda_pair_is_constrained_in_package_contract():
    pins = {dist: version for dist, _, version in PACKAGES}
    assert pins["torch"] == "2.11.0+cu128"
    assert pins["torchvision"] == "0.26.0+cu128"
