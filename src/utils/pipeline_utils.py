#!/usr/bin/env python3
"""
パイプライン関連ユーティリティ関数
"""

import numpy as np
import pandas as pd


def get_pipeline_feature_names(pipeline, original_feature_names):
    """
    パイプライン変換後の正しい特徴量名を取得

    Args:
        pipeline: sklearn Pipeline or ImbPipeline
        original_feature_names: 元の特徴量名のリスト

    Returns:
        list: パイプライン変換後の特徴量名
    """
    # サンプリングクラス（テスト時はスキップ）
    SAMPLING_CLASSES = [
        'SMOTE', 'RandomOverSampler', 'RandomUnderSampler',
        'ADASYN', 'BorderlineSMOTE', 'SVMSMOTE'
    ]

    # 最終ステップ（分類器・回帰器）を除く変換ステップを取得
    transform_steps = pipeline.steps[:-1]

    # 現在の特徴量名を初期化
    current_feature_names = list(original_feature_names)

    # 各変換ステップを順次適用
    for step_name, transformer in transform_steps:
        # サンプリング処理はテスト時にスキップ（特徴量名は変更されない）
        if any(cls in str(type(transformer)) for cls in SAMPLING_CLASSES):
            continue

        # get_feature_names_out()メソッドが存在する場合
        if hasattr(transformer, 'get_feature_names_out'):
            try:
                current_feature_names = transformer.get_feature_names_out(current_feature_names)
                if hasattr(current_feature_names, 'tolist'):
                    current_feature_names = current_feature_names.tolist()
            except Exception as e:
                print(f"⚠️ {step_name}のget_feature_names_out()でエラー: {e}")
                # エラーの場合は変更せずに継続
                continue

        # StandardScalerなどsklearnの標準Transformerの場合
        elif hasattr(transformer, 'feature_names_in_'):
            # 特徴量数が変わらない場合はそのまま
            continue

        else:
            print(f"⚠️ {step_name}は特徴量名の取得方法が不明（スキップ）")
            continue

    return list(current_feature_names)


def get_transformed_data_with_feature_names(pipeline, X, original_feature_names):
    """
    パイプライン変換後のデータと正しい特徴量名を取得

    Args:
        pipeline: sklearn Pipeline or ImbPipeline
        X: 入力データ
        original_feature_names: 元の特徴量名

    Returns:
        tuple: (変換後データ, 変換後特徴量名)
    """
    # サンプリングクラス（テスト時はスキップ）
    SAMPLING_CLASSES = [
        'SMOTE', 'RandomOverSampler', 'RandomUnderSampler',
        'ADASYN', 'BorderlineSMOTE', 'SVMSMOTE'
    ]

    # パイプライン変換（最終ステップ前まで）
    try:
        # ImbPipelineまたは通常のPipelineでのtransform処理
        X_transformed = pipeline[:-1].transform(X)
    except AttributeError:
        # サンプラーが含まれる場合の手動変換
        X_current = X.copy()
        for step_name, transformer in pipeline.steps[:-1]:
            # サンプラーはtransformメソッドを持たないため、テスト時はスキップ
            if any(cls in str(type(transformer)) for cls in SAMPLING_CLASSES):
                continue
            X_current = transformer.transform(X_current)
        X_transformed = X_current

    # 変換後の特徴量名を取得
    transformed_feature_names = get_pipeline_feature_names(pipeline, original_feature_names)

    return X_transformed, transformed_feature_names