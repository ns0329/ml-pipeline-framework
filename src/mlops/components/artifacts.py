"""MLflowアーティファクト管理コンポーネント"""
import tempfile
import joblib
import json
import os
import mlflow


def save_model_artifacts(pipeline, feature_names, target_names, cfg):
    """モデルと関連情報をMLflowに保存（config駆動、一時ディレクトリ処理統一）"""
    with tempfile.TemporaryDirectory() as tmp_dir:
        # モデル保存（config化）
        model_path = os.path.join(tmp_dir, cfg.mlflow.artifacts.model_filename)
        joblib.dump(pipeline, model_path)
        mlflow.log_artifact(model_path, cfg.mlflow.artifacts.model_dir)

        # モデル情報保存（config化）
        info_path = os.path.join(tmp_dir, cfg.mlflow.artifacts.info_filename)
        with open(info_path, 'w') as f:
            json.dump({
                "model_type": type(pipeline.named_steps['classifier']).__name__,
                "feature_names": list(feature_names),
                "target_names": [str(x) for x in target_names]
            }, f)
        mlflow.log_artifact(info_path, cfg.mlflow.artifacts.model_dir)


def log_config_parameters(cfg):
    """実験設定全体をMLflowに記録（最短版）"""
    from omegaconf import OmegaConf
    for key, value in OmegaConf.to_container(cfg, resolve=True).items():
        mlflow.log_param(key, str(value))


def log_runtime_parameters(pipeline, cfg, best_params=None):
    """実行時パラメータをMLflowに記録（パイプライン、モデルパラメータ）"""
    # パイプライン構成情報
    mlflow.log_param("pipeline_steps", len(pipeline.steps))
    for i, (step_name, step_obj) in enumerate(pipeline.steps):
        mlflow.log_param(f"step_{i}_name", step_name)
        mlflow.log_param(f"step_{i}_class", type(step_obj).__name__)

    # モデルパラメータ（常に記録）
    model = pipeline.named_steps.get('classifier')
    if model:
        # モデルの実際のパラメータを記録
        model_params = model.get_params()
        for key, value in model_params.items():
            mlflow.log_param(f"model_{key}", str(value))

        # Optuna最適化情報
        mlflow.log_param("tuning_optimized", best_params is not None)
        if best_params:
            for key, value in best_params.items():
                mlflow.log_param(f"tuned_{key}", str(value))


def log_experiment_metrics(best_pipeline, X_train, y_train, X_test, y_test, task_type, cv_scores, cfg=None, y_pred=None):
    """実験メトリクスをMLflowに記録"""
    import numpy as np
    from sklearn.metrics import (
        accuracy_score, f1_score, roc_auc_score, precision_score, recall_score,
        mean_squared_error, mean_absolute_error, r2_score
    )

    # CV結果記録
    mlflow.log_metric("cv_mean", cv_scores.mean())
    mlflow.log_metric("cv_std", cv_scores.std())

    # Test評価（予測結果が渡されない場合のみ実行）
    if y_pred is None:
        y_pred = best_pipeline.predict(X_test)

    if task_type == "classification":
        # 分類評価指標
        test_accuracy = accuracy_score(y_test, y_pred)
        mlflow.log_metric("test_accuracy", test_accuracy)

        # config設定に基づくメトリクス計算
        scoring_type = cfg.optuna.scoring.classification if cfg else "f1_weighted"
        is_binary = len(np.unique(y_test)) == 2

        # F1スコア計算（config設定に対応）
        if scoring_type in ["f1", "f1_binary"] and is_binary:
            # 二値分類: binaryモード（陽性クラスのF1）
            test_f1 = f1_score(y_test, y_pred, pos_label=1)
            metric_suffix = "binary"
        elif scoring_type == "f1_macro":
            test_f1 = f1_score(y_test, y_pred, average='macro')
            metric_suffix = "macro"
        else:  # f1_weighted or default
            test_f1 = f1_score(y_test, y_pred, average='weighted')
            metric_suffix = "weighted"

        mlflow.log_metric(f"test_f1_{metric_suffix}", test_f1)

        # AUC（二値・多値分類対応）
        if hasattr(best_pipeline, 'predict_proba'):
            y_proba = best_pipeline.predict_proba(X_test)
            if is_binary:  # 二値分類
                test_auc = roc_auc_score(y_test, y_proba[:, 1])
            else:  # 多値分類
                test_auc = roc_auc_score(y_test, y_proba, multi_class='ovr', average='weighted')
            mlflow.log_metric("test_auc", test_auc)

        # Precision/Recall（configと同じaverageを使用）
        if scoring_type in ["f1", "f1_binary", "precision", "recall"] and is_binary:
            test_precision = precision_score(y_test, y_pred, pos_label=1)
            test_recall = recall_score(y_test, y_pred, pos_label=1)
            mlflow.log_metric("test_precision_binary", test_precision)
            mlflow.log_metric("test_recall_binary", test_recall)
        elif "macro" in scoring_type:
            test_precision = precision_score(y_test, y_pred, average='macro')
            test_recall = recall_score(y_test, y_pred, average='macro')
            mlflow.log_metric("test_precision_macro", test_precision)
            mlflow.log_metric("test_recall_macro", test_recall)
        else:
            test_precision = precision_score(y_test, y_pred, average='weighted')
            test_recall = recall_score(y_test, y_pred, average='weighted')
            mlflow.log_metric("test_precision_weighted", test_precision)
            mlflow.log_metric("test_recall_weighted", test_recall)

        # 表示用ラベル
        f1_label = "F1" if metric_suffix == "binary" else f"F1_{metric_suffix[:3]}"
        print(f"🚀 {mlflow.active_run().info.run_id[:8]} | 📈 CV: {cv_scores.mean():.3f}±{cv_scores.std():.3f} Test Acc: {test_accuracy:.3f} {f1_label}: {test_f1:.3f} AUC: {test_auc:.3f}")

    else:  # regression
        # 回帰評価指標
        test_mse = mean_squared_error(y_test, y_pred)
        test_rmse = np.sqrt(test_mse)
        test_mae = mean_absolute_error(y_test, y_pred)
        test_r2 = r2_score(y_test, y_pred)

        mlflow.log_metric("test_mse", test_mse)
        mlflow.log_metric("test_rmse", test_rmse)
        mlflow.log_metric("test_mae", test_mae)
        mlflow.log_metric("test_r2", test_r2)

        # Train評価（回帰では比較用）
        y_train_pred = best_pipeline.predict(X_train)
        train_r2 = r2_score(y_train, y_train_pred)
        mlflow.log_metric("train_r2", train_r2)

        print(f"🚀 {mlflow.active_run().info.run_id[:8]} | 📈 CV: {cv_scores.mean():.3f}±{cv_scores.std():.3f} Test RMSE: {test_rmse:.3f} R²: {test_r2:.3f}")


def create_prediction_dataframe(pipeline, X_test, y_test, task_type, y_pred=None):
    """予測結果と変換済み特徴量を統合したDataFrameを作成"""
    import pandas as pd
    import numpy as np

    # 予測値と予測確率を取得（予測結果が渡されない場合のみ実行）
    if y_pred is None:
        y_pred = pipeline.predict(X_test)

    # パイプライン変換後の特徴量を取得（最終ステップ前まで）
    try:
        # ImbPipelineまたは通常のPipelineでのtransform処理
        X_test_transformed = pipeline[:-1].transform(X_test)
    except AttributeError:
        # サンプラーが含まれる場合の手動変換
        from src.mlops.components.pipeline import SAMPLING_CLASSES
        X_current = X_test.copy()
        for name, transformer in pipeline.steps[:-1]:
            # サンプラーはtransformメソッドを持たないため、テスト時はスキップ
            if any(cls in str(type(transformer)) for cls in SAMPLING_CLASSES):
                continue
            X_current = transformer.transform(X_current)
        X_test_transformed = X_current

    # 変換後データをDataFrameに変換
    if isinstance(X_test_transformed, pd.DataFrame):
        df_transformed = X_test_transformed.copy()
    else:
        # numpy arrayの場合（パイプライン変換後の正しい特徴量名を使用）
        from src.utils.pipeline_utils import get_pipeline_feature_names
        original_feature_names = (X_test.columns.tolist()
                                if hasattr(X_test, 'columns')
                                else [f"feature_{i}" for i in range(X_test.shape[1])])
        transformed_feature_names = get_pipeline_feature_names(pipeline, original_feature_names)

        n_features = X_test_transformed.shape[1]
        # 変換後の特徴量数に合わせて調整
        if len(transformed_feature_names) >= n_features:
            feature_names = transformed_feature_names[:n_features]
        else:
            feature_names = transformed_feature_names + [f"generated_feature_{i}" for i in range(len(transformed_feature_names), n_features)]

        df_transformed = pd.DataFrame(X_test_transformed, columns=feature_names, index=X_test.index)

    # 元のテストデータとマージ（重複カラムはパイプライン側を優先）
    df_result = X_test.copy()

    # パイプライン変換後の特徴量で上書き
    for col in df_transformed.columns:
        if col in df_result.columns:
            # 重複カラムはパイプライン側を採用
            df_result[col] = df_transformed[col].values
        else:
            # 新規生成カラムを追加
            df_result[f"generated_{col}"] = df_transformed[col].values

    # 実際の値と予測値を追加
    df_result['y_true'] = y_test.values
    df_result['y_pred'] = y_pred

    # タスクタイプに応じて予測確率を追加
    if task_type == "classification" and hasattr(pipeline, 'predict_proba'):
        y_proba = pipeline.predict_proba(X_test)
        n_classes = y_proba.shape[1]

        # 各クラスの予測確率を追加
        for i in range(n_classes):
            df_result[f'proba_class_{i}'] = y_proba[:, i]

        # 予測の信頼度（最大確率）
        df_result['prediction_confidence'] = y_proba.max(axis=1)

    return df_result


def save_prediction_results(df_predictions, cfg):
    """予測結果DataFrameをMLflowアーティファクトとして保存"""
    import tempfile
    import os

    with tempfile.TemporaryDirectory() as tmp_dir:
        # CSV保存
        csv_path = os.path.join(tmp_dir, "test_predictions.csv")
        df_predictions.to_csv(csv_path, index=False)

        # MLflowに保存
        mlflow.log_artifact(csv_path, "predictions")

        print(f"💾 予測結果CSV保存完了: predictions/test_predictions.csv")
        print(f"   - データ件数: {len(df_predictions)}件")
        print(f"   - カラム数: {len(df_predictions.columns)}列")

    return df_predictions


def setup_mlflow_experiment(cfg):
    """MLflow実験セットアップ"""
    mlflow.set_experiment(cfg.mlflow.experiment_name)


def set_mlflow_tags(cfg):
    """MLflowタグ設定（run開始後に呼び出し）"""
    if hasattr(cfg.mlflow, 'tags'):
        for key, value in cfg.mlflow.tags.items():
            mlflow.set_tag(key, value)