import numpy as np
import pandas as pd

from spaq_iqa.exif import EXIF_COLUMNS, ExifPreprocessor


def frame(rows: list[list[float]]) -> pd.DataFrame:
    return pd.DataFrame(rows, columns=EXIF_COLUMNS)


def test_missing_values_have_zero_value_and_false_mask() -> None:
    train = frame([[4, 2, 0.01, 100, 1, 0, -1, 0], [8, 4, 0.1, 400, 3, 1, 0, 1]])
    processor = ExifPreprocessor().fit(train)
    test = train.iloc[[0]].copy()
    test["ISO"] = np.nan
    values, mask = processor.transform(test)
    iso_index = EXIF_COLUMNS.index("ISO")
    assert values[0, iso_index] == 0
    assert mask[0, iso_index] == 0
    assert mask.sum() == len(EXIF_COLUMNS) - 1


def test_scaler_state_round_trip_and_train_only_fit() -> None:
    train = frame([[4, 2, 0.01, 100, 1, 0, -1, 0], [8, 4, 0.1, 400, 3, 1, 0, 1]])
    validation = frame([[40, 20, 10, 6400, 30, 0, 1, 0]])
    processor = ExifPreprocessor().fit(train)
    before = processor.state_dict()
    transformed, _ = processor.transform(validation)
    restored = ExifPreprocessor.from_state_dict(before)
    transformed_again, _ = restored.transform(validation)
    np.testing.assert_allclose(transformed, transformed_again)
    assert processor.state_dict() == before


def test_negative_positive_domain_value_is_marked_missing() -> None:
    train = frame([[4, 2, 0.01, 100, 1, 0, -1, 0], [8, 4, 0.1, 400, 3, 1, 0, 1]])
    invalid = train.iloc[[0]].copy()
    invalid.loc[:, "Exposure time"] = -1
    values, mask = ExifPreprocessor().fit(train).transform(invalid)
    index = EXIF_COLUMNS.index("Exposure time")
    assert values[0, index] == 0
    assert mask[0, index] == 0
