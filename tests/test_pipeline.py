
import pandas as pd
import pytest
import joblib
import yaml

from src.data import validate, preprocess


def create_sample_dataset(path):
    """Create a temporary dataset for testing."""
    df = pd.DataFrame({
        "feature_1": range(100),
        "feature_2": range(100, 200),
        "feature_3": range(200, 300),
        "target": [0, 1] * 50
    })
    df.to_csv(path, index=False)


def test_validate_data_success(tmp_path, monkeypatch):
    raw_file = tmp_path / "sample.csv"
    report_file = tmp_path / "validation.txt"

    create_sample_dataset(raw_file)

    monkeypatch.setattr(validate, "RAW_DATA_PATH", raw_file)
    monkeypatch.setattr(validate, "REPORT_PATH", report_file)

    assert validate.validate_data() is True
    assert report_file.exists()
    assert "PASSED" in report_file.read_text()


def test_validate_data_missing_target(tmp_path, monkeypatch):
    raw_file = tmp_path / "invalid.csv"

    pd.DataFrame({
        "feature_1": [1, 2, 3]
    }).to_csv(raw_file, index=False)

    monkeypatch.setattr(validate, "RAW_DATA_PATH", raw_file)

    with pytest.raises(ValueError, match="Missing target column"):
        validate.validate_data()


def test_preprocess_data_success(tmp_path, monkeypatch):
    raw_file = tmp_path / "sample.csv"
    processed_dir = tmp_path / "processed"
    params_file = tmp_path / "params.yaml"

    create_sample_dataset(raw_file)

    params = {
        "data": {
            "test_size": 0.2,
            "random_state": 42
        }
    }

    params_file.write_text(yaml.safe_dump(params))

    monkeypatch.setattr(preprocess, "RAW_DATA_PATH", raw_file)
    monkeypatch.setattr(preprocess, "PROCESSED_DIR", processed_dir)
    monkeypatch.setattr(preprocess, "PARAMS_PATH", params_file)

    preprocess.preprocess_data()

    train_file = processed_dir / "train.csv"
    test_file = processed_dir / "test.csv"
    scaler_file = processed_dir / "scaler.joblib"

    assert train_file.exists()
    assert test_file.exists()
    assert scaler_file.exists()

    train_df = pd.read_csv(train_file)
    test_df = pd.read_csv(test_file)

    assert len(train_df) == 80
    assert len(test_df) == 20

    assert "target" in train_df.columns
    assert "target" in test_df.columns

    scaler = joblib.load(scaler_file)
    assert hasattr(scaler, "mean_")
