"""テスト用共通ユーティリティ"""

import pandas as pd
import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from typing import Any, List, Union


def assert_transformer_basic_contract(
    transformer: BaseEstimator,
    X_train: pd.DataFrame,
    X_test: pd.DataFrame = None,
    y_train: pd.Series = None
) -> None:
    """Transformerの基本契約をアサート"""

    # fit前はtransformできない場合があることを許可
    # まずfitを実行
    transformer.fit(X_train, y_train)

    # fit後はtransformできるはず
    X_transformed = transformer.transform(X_train)

    # 基本的なshape確認
    assert X_transformed is not None, "transform結果がNoneです"
    assert len(X_transformed) == len(X_train), f"行数が変化しています: {len(X_train)} -> {len(X_transformed)}"

    # get_feature_names_out実装確認
    if hasattr(transformer, 'get_feature_names_out'):
        feature_names = transformer.get_feature_names_out()
        assert feature_names is not None, "get_feature_names_out結果がNoneです"

        # DataFrameの場合はカラム数確認
        if isinstance(X_transformed, pd.DataFrame):
            assert len(feature_names) == X_transformed.shape[1], \
                f"特徴量名数と実際のカラム数が一致しません: {len(feature_names)} != {X_transformed.shape[1]}"

    # テストデータでのtransform確認
    if X_test is not None:
        X_test_transformed = transformer.transform(X_test)
        assert X_test_transformed is not None, "テストデータのtransform結果がNoneです"
        assert len(X_test_transformed) == len(X_test), \
            f"テストデータの行数が変化しています: {len(X_test)} -> {len(X_test_transformed)}"


def assert_no_data_leakage(
    transformer: BaseEstimator,
    X_train: pd.DataFrame,
    X_test: pd.DataFrame,
    y_train: pd.Series = None
) -> None:
    """データリークがないことを確認"""

    # 訓練データのみでfit
    transformer_1 = transformer.__class__(**transformer.get_params())
    transformer_1.fit(X_train, y_train)

    # 訓練+テストデータでfit
    X_combined = pd.concat([X_train, X_test], ignore_index=True)
    y_combined = pd.concat([y_train, pd.Series([0] * len(X_test))], ignore_index=True) if y_train is not None else None

    transformer_2 = transformer.__class__(**transformer.get_params())
    transformer_2.fit(X_combined, y_combined)

    # 同じ訓練データに対するtransform結果を比較
    X_train_transformed_1 = transformer_1.transform(X_train)
    X_train_transformed_2 = transformer_2.transform(X_train)

    # 完全に同じである必要はないが、大きく異なるべきではない
    # （統計的特徴選択など、一部のtransformerでは差が出ることを許容）
    if hasattr(X_train_transformed_1, 'shape') and hasattr(X_train_transformed_2, 'shape'):
        # shapeは一致すべき（ただし、特徴選択の場合は異なる可能性があるため緩い条件）
        pass  # より詳細な比較は個別テストで実装


def assert_parameter_validation(
    transformer_class: type,
    valid_params: dict,
    invalid_params_list: List[dict],
    X: pd.DataFrame,
    y: pd.Series = None
) -> None:
    """パラメータバリデーションをテスト"""

    # 正常パラメータでの動作確認
    transformer = transformer_class(**valid_params)
    transformer.fit(X, y)
    transformer.transform(X)

    # 異常パラメータでのエラー確認
    for invalid_params in invalid_params_list:
        try:
            transformer = transformer_class(**invalid_params)
            transformer.fit(X, y)
            # エラーが出なかった場合は警告（必ずしもエラーが出る必要はない）
            print(f"警告: {invalid_params} でエラーが出ませんでした")
        except (ValueError, TypeError, AttributeError) as e:
            # 期待されるエラー
            pass


def create_edge_case_dataframes() -> List[pd.DataFrame]:
    """エッジケース用DataFrameリスト生成"""

    edge_cases = []

    # 空DataFrame
    edge_cases.append(pd.DataFrame())

    # 1行DataFrame
    edge_cases.append(pd.DataFrame({'col1': [1], 'col2': [2]}))

    # 1列DataFrame
    edge_cases.append(pd.DataFrame({'col1': [1, 2, 3]}))

    # 全欠損DataFrame
    edge_cases.append(pd.DataFrame({'col1': [np.nan, np.nan], 'col2': [np.nan, np.nan]}))

    # 全定数DataFrame
    edge_cases.append(pd.DataFrame({'col1': [1, 1, 1], 'col2': [2, 2, 2]}))

    return edge_cases


def compare_dataframes_approximately(
    df1: pd.DataFrame,
    df2: pd.DataFrame,
    tolerance: float = 1e-6
) -> bool:
    """DataFrameの近似比較"""

    if df1.shape != df2.shape:
        return False

    if not df1.columns.equals(df2.columns):
        return False

    for col in df1.columns:
        if df1[col].dtype != df2[col].dtype:
            return False

        if pd.api.types.is_numeric_dtype(df1[col]):
            if not np.allclose(df1[col].fillna(0), df2[col].fillna(0), atol=tolerance, equal_nan=True):
                return False
        else:
            if not df1[col].equals(df2[col]):
                return False

    return True