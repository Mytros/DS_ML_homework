import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from typing import Tuple, List, Union


def split_data(df: pd.DataFrame, target_col: str) -> Tuple[pd.DataFrame, pd.Series, pd.DataFrame, pd.Series]:
    """
    Split the dataset into train and validation sets.

    Args:
        df (pd.DataFrame): Raw dataset.
        target_col (str): Name of the target column.

    Returns:
        Tuple of X_train, y_train, X_val, y_val.
    """
    train_df, val_df = train_test_split(df, test_size=0.2, random_state=42, stratify=df[target_col])
    X_train = train_df.drop(columns=[target_col])
    y_train = train_df[target_col]
    X_val = val_df.drop(columns=[target_col])
    y_val = val_df[target_col]
    return X_train, y_train, X_val, y_val


def encode_categorical(
    X_train: pd.DataFrame, X_val: pd.DataFrame, cat_cols: List[str]
) -> Tuple[pd.DataFrame, pd.DataFrame, OneHotEncoder]:
    """
    One-hot encode categorical columns.

    Args:
        X_train: Training features.
        X_val: Validation features.
        cat_cols: Categorical column names.

    Returns:
        Encoded X_train, X_val, and fitted encoder.
    """
    encoder = OneHotEncoder(handle_unknown='ignore', sparse_output=False)
    encoder.fit(X_train[cat_cols])

    train_encoded = pd.DataFrame(
        encoder.transform(X_train[cat_cols]),
        columns=encoder.get_feature_names_out(cat_cols),
        index=X_train.index
    )
    val_encoded = pd.DataFrame(
        encoder.transform(X_val[cat_cols]),
        columns=encoder.get_feature_names_out(cat_cols),
        index=X_val.index
    )

    X_train = X_train.drop(columns=cat_cols)
    X_val = X_val.drop(columns=cat_cols)

    return pd.concat([X_train, train_encoded], axis=1), pd.concat([X_val, val_encoded], axis=1), encoder


def scale_numeric(
    X_train: pd.DataFrame, X_val: pd.DataFrame, num_cols: List[str]
) -> Tuple[pd.DataFrame, pd.DataFrame, StandardScaler]:
    """
    Scale numeric features with StandardScaler.

    Args:
        X_train: Training features.
        X_val: Validation features.
        num_cols: List of numeric column names.

    Returns:
        Scaled X_train, X_val, and fitted scaler.
    """
    scaler = StandardScaler()
    scaler.fit(X_train[num_cols])
    X_train[num_cols] = scaler.transform(X_train[num_cols])
    X_val[num_cols] = scaler.transform(X_val[num_cols])
    return X_train, X_val, scaler


def preprocess_data(
    raw_df: pd.DataFrame,
    scaler_numeric: bool = False
) -> Tuple[pd.DataFrame, pd.Series, pd.DataFrame, pd.Series, List[str], Union[StandardScaler, None], OneHotEncoder]:
    """
    Preprocess the dataset for training decision trees or other models.

    Args:
        raw_df: Raw input DataFrame.
        scaler_numeric: Whether to apply numeric feature scaling.

    Returns:
        X_train, y_train, X_val, y_val, input feature names, scaler, encoder.
    """
    df = raw_df.copy()
    df = df.drop(columns=["Surname"])  # Drop non-informative column

    target_col = "Exited"
    X_train, y_train, X_val, y_val = split_data(df, target_col)

    cat_cols = X_train.select_dtypes(include="object").columns.tolist()
    num_cols = X_train.select_dtypes(include=np.number).columns.tolist()

    X_train, X_val, encoder = encode_categorical(X_train, X_val, cat_cols)

    scaler = None
    if scaler_numeric:
        X_train, X_val, scaler = scale_numeric(X_train, X_val, num_cols)

    input_cols = X_train.columns.tolist()

    return X_train, y_train, X_val, y_val, input_cols, scaler, encoder


def preprocess_new_data(
    df: pd.DataFrame,
    input_cols: List[str],
    encoder: OneHotEncoder,
    scaler: Union[StandardScaler, None] = None
) -> pd.DataFrame:
    """
    Preprocess new data using fitted encoder and scaler.

    Args:
        df: Raw new data.
        input_cols: Expected input columns after preprocessing.
        encoder: Fitted OneHotEncoder.
        scaler: Optional fitted StandardScaler.

    Returns:
        Processed DataFrame with the same columns as training data.
    """
    df = df.copy()
    df = df.drop(columns=["Surname"], errors="ignore")

    cat_cols = df.select_dtypes(include="object").columns.tolist()
    num_cols = df.select_dtypes(include=np.number).columns.tolist()

    encoded = pd.DataFrame(
        encoder.transform(df[cat_cols]),
        columns=encoder.get_feature_names_out(cat_cols),
        index=df.index
    )
    df = df.drop(columns=cat_cols)
    df = pd.concat([df, encoded], axis=1)

    if scaler:
        df[num_cols] = scaler.transform(df[num_cols])

    return df.reindex(columns=input_cols, fill_value=0)
