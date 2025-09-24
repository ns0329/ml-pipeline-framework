"""
MLOpsパイプライン関連ユーティリティ

scikit-learnパイプライン変換後の特徴量名追跡と
データ変換処理を提供する。
"""

from typing import List, Tuple, Any
import numpy as np
import pandas as pd


def get_pipeline_feature_names(pipeline, original_feature_names: List[str]) -> List[str]:
    """
    パイプライン変換後の正しい特徴量名を取得する。

    Args:
        pipeline: sklearn Pipeline or ImbPipeline
        original_feature_names: 元の特徴量名のリスト

    Returns:
        パイプライン変換後の特徴量名のリスト

    Note:
        サンプリングクラス（SMOTE等）はテスト時にスキップされ、
        特徴量名は変更されない。
    """
    SAMPLING_CLASSES = [
        'SMOTE', 'RandomOverSampler', 'RandomUnderSampler',
        'ADASYN', 'BorderlineSMOTE', 'SVMSMOTE'
    ]

    transform_steps = pipeline.steps[:-1]  # 最終ステップ（分類器・回帰器）を除く
    current_feature_names = list(original_feature_names)

    for step_name, transformer in transform_steps:
        if any(cls in str(type(transformer)) for cls in SAMPLING_CLASSES):
            continue  # サンプリング処理はテスト時にスキップ

        if hasattr(transformer, 'get_feature_names_out'):
            try:
                current_feature_names = transformer.get_feature_names_out(current_feature_names)
                if hasattr(current_feature_names, 'tolist'):
                    current_feature_names = current_feature_names.tolist()
            except Exception as e:
                print(f"⚠️ {step_name}のget_feature_names_out()でエラー: {e}")
                continue

        elif hasattr(transformer, 'feature_names_in_'):
            continue  # StandardScaler等、特徴量数が変わらない場合

        else:
            print(f"⚠️ {step_name}は特徴量名の取得方法が不明（スキップ）")
            continue

    return list(current_feature_names)


def get_transformed_data_with_feature_names(
    pipeline, X: Any, original_feature_names: List[str]
) -> Tuple[Any, List[str]]:
    """
    パイプライン変換後のデータと正しい特徴量名を取得する。

    Args:
        pipeline: sklearn Pipeline or ImbPipeline
        X: 入力データ
        original_feature_names: 元の特徴量名

    Returns:
        (変換後データ, 変換後特徴量名)のタプル
    """
    SAMPLING_CLASSES = [
        'SMOTE', 'RandomOverSampler', 'RandomUnderSampler',
        'ADASYN', 'BorderlineSMOTE', 'SVMSMOTE'
    ]

    try:
        X_transformed = pipeline[:-1].transform(X)
    except AttributeError:
        # サンプラーが含まれる場合の手動変換
        X_current = X.copy()
        for step_name, transformer in pipeline.steps[:-1]:
            if any(cls in str(type(transformer)) for cls in SAMPLING_CLASSES):
                continue
            X_current = transformer.transform(X_current)
        X_transformed = X_current

    transformed_feature_names = get_pipeline_feature_names(pipeline, original_feature_names)
    return X_transformed, transformed_feature_names