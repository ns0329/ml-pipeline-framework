"""MLOps実験実行スクリプト（リファクタリング版）"""
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score
import mlflow
import hydra
from omegaconf import DictConfig

# データ処理ユーティリティ
from src.utils.data_utils import get_dataset_name, detect_task_type, load_csv_data
from src.utils.cv_utils import create_cv_strategy

# components機能インポート
from src.mlops.components.pipeline import create_pipeline
from src.mlops.components.visualization import create_visualizations
from src.mlops.components.optimization import OptunaOptimizer
from src.mlops.components.artifacts import save_model_artifacts, log_experiment_metrics, setup_mlflow_experiment, set_mlflow_tags, log_config_parameters, log_runtime_parameters, create_prediction_dataframe, save_prediction_results

# matplotlib設定
import os
import logging
os.environ['MPLBACKEND'] = 'Agg'

# matplotlibフォント警告抑制
logging.getLogger('matplotlib.font_manager').setLevel(logging.ERROR)

import matplotlib
# DejaVu Sansフォント設定
matplotlib.rcParams['font.family'] = 'DejaVu Sans'
matplotlib.rcParams['font.sans-serif'] = ['DejaVu Sans', 'Ubuntu']

# 日本語フォント対応
try:
    import japanize_matplotlib
except ImportError:
    pass

# 一般的な警告抑制
import warnings
warnings.filterwarnings('ignore')


@hydra.main(version_base=None, config_path="../../config", config_name="config")
def main(cfg: DictConfig):
    """メイン実行関数"""
    print("✅ Config駆動MLOpsフロー開始")

    # MLflow実験セットアップ
    setup_mlflow_experiment(cfg)

    # 既存runがある場合は終了
    if mlflow.active_run():
        mlflow.end_run()

    # カスタムRun名設定（オプション）
    run_name = getattr(cfg.mlflow, 'run_id', None)

    with mlflow.start_run(run_name=run_name):
        # タグ設定（run開始後）
        set_mlflow_tags(cfg)

        # データ読み込み
        df, feature_cols, target_names = load_csv_data(cfg)

        # データ分割
        X = df[feature_cols]
        y = df[cfg.data.target_column]
        X_train, X_test, y_train, y_test = train_test_split(
            X, y,
            test_size=cfg.data.test_size,
            random_state=cfg.data.random_state
        )

        print(f"🔍 Task Type: {detect_task_type(y)}")
        task_type = detect_task_type(y)
        print(f"🔍 X_train type: {type(X_train)}, shape: {X_train.shape}, columns: {list(X_train.columns)}")

        # Optuna最適化（有効な場合）
        if cfg.optuna.enabled:
            optimizer = OptunaOptimizer(cfg, X_train, y_train, task_type)
            best_params, best_score = optimizer.optimize()
            print(f"🎯 Optuna best_params: {best_params}")
        else:
            best_params = {}
            print(f"⚠️ Optuna無効: best_params = {best_params}")

        # 最適化されたパイプライン構築（best_paramsを反映）
        passed_params = best_params if best_params else None
        print(f"📦 create_pipeline呼び出し: best_params={passed_params}")
        best_pipeline = create_pipeline(cfg, best_params=passed_params)

        # パイプライン情報表示
        print(f"🔧 Pipeline: {best_pipeline} | 📊 Train: {len(X_train)} Test: {len(X_test)}")

        # パイプライン学習
        best_pipeline.fit(X_train, y_train)

        # 実行時パラメータ記録（パイプライン、最適化結果など）
        log_runtime_parameters(best_pipeline, cfg, best_params)

        # テストデータ予測（1回のみ実行）
        y_pred = best_pipeline.predict(X_test)

        # Optuna最適化時はCV評価済み、未実行時のみCV実行
        if not cfg.optuna.enabled:
            # クロスバリデーション評価（Optuna未使用時のみ）
            if task_type == "classification":
                scoring = cfg.optuna.scoring.classification
            else:
                scoring = cfg.optuna.scoring.regression

            cv_strategy = create_cv_strategy(cfg)
            print(f"🔄 CV Strategy: {cfg.evaluation.cv_strategy['class']} (n_splits={cfg.evaluation.cv_strategy.params.n_splits})")

            cv_scores = cross_val_score(
                best_pipeline, X_train, y_train,
                cv=cv_strategy,
                scoring=scoring
            )
        else:
            # Optuna使用時は最適化結果を使用
            cv_scores = np.array([best_score] * 5)  # best_scoreを5foldに展開（numpy配列で互換性維持）
            print(f"🔄 CV評価をスキップ（Optuna最適化済み: {best_score:.3f}）")

        # メトリクス記録
        log_experiment_metrics(best_pipeline, X_train, y_train, X_test, y_test, task_type, cv_scores, y_pred=y_pred)

        # 予測結果DataFrame作成と保存
        df_predictions = create_prediction_dataframe(best_pipeline, X_test, y_test, task_type, y_pred=y_pred)
        save_prediction_results(df_predictions, cfg)

        # 可視化生成
        if cfg.visualization.enabled:
            target_names_str = [str(name) for name in target_names]
            create_visualizations(
                best_pipeline, X_train, y_train, X_test, y_test,
                target_names_str, cfg.visualization.plots, cfg, task_type
            )

        # モデル・アーティファクト保存（パイプライン変換後の特徴量名を使用）
        from src.utils.pipeline_utils import get_pipeline_feature_names
        transformed_feature_names = get_pipeline_feature_names(best_pipeline, feature_cols)
        save_model_artifacts(best_pipeline, transformed_feature_names, target_names, cfg)

        print("✅ MLOpsフロー完了")


if __name__ == "__main__":
    main()