import numpy as np
import pytest

from spaq_iqa.metrics import regression_metrics


def test_regression_metrics_known_arrays() -> None:
    result = regression_metrics(np.array([1, 2, 3, 4]), np.array([2, 4, 6, 8]))
    assert result["srcc"] == pytest.approx(1.0)
    assert result["plcc"] == pytest.approx(1.0)
    assert result["mae"] == pytest.approx(2.5)


def test_regression_metrics_reject_nonfinite() -> None:
    with pytest.raises(ValueError):
        regression_metrics(np.array([1.0, np.nan]), np.array([1.0, 2.0]))
