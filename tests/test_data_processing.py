import pandas as pd


def test_customer_id_exists():

    df = pd.read_csv(
        "data/processed/processed_data.csv"
    )

    assert "CustomerId" in df.columns


def test_target_exists():

    df = pd.read_csv(
        "data/processed/processed_data.csv"
    )

    assert "is_high_risk" in df.columns
