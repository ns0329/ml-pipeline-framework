"""可視化・解釈コンポーネント"""
import tempfile
import os
import logging
import japanize_matplotlib
from abc import ABC, abstractmethod
from typing import Dict, List, Any, Optional

# matplotlib設定
import matplotlib
matplotlib.use('Agg')  # バックエンド設定
# フォント警告抑制とフォント設定
logging.getLogger('matplotlib.font_manager').setLevel(logging.ERROR)
matplotlib.rcParams['font.family'] = 'DejaVu Sans'
matplotlib.rcParams['font.sans-serif'] = ['DejaVu Sans', 'Ubuntu']

import matplotlib.pyplot as plt
import numpy as np
import mlflow


# ============================================================================
# 抽象ベースクラス
# ============================================================================

class BaseVisualizer(ABC):
    """可視化の抽象ベースクラス"""

    def __init__(self, pipeline, X_train, y_train, X_test, y_test, task_type: str):
        self.pipeline = pipeline
        self.X_train = X_train
        self.y_train = y_train
        self.X_test = X_test
        self.y_test = y_test
        self.task_type = task_type
        self.model = pipeline.named_steps['classifier']

        # データ変換（ImbPipeline対応）
        try:
            # ImbPipelineまたは通常のPipelineでのtransform処理
            self.X_train_transformed = pipeline[:-1].transform(X_train)
            self.X_test_transformed = pipeline[:-1].transform(X_test)
        except AttributeError:
            # サンプラーが含まれる場合の手動変換
            from src.mlops.components.pipeline import SAMPLING_CLASSES

            # X_train変換
            X_current = X_train.copy()
            for name, transformer in pipeline.steps[:-1]:
                # サンプラーはtransformメソッドを持たないため、学習時はスキップ
                if any(cls in str(type(transformer)) for cls in SAMPLING_CLASSES):
                    continue
                X_current = transformer.transform(X_current)
            self.X_train_transformed = X_current

            # X_test変換
            X_current = X_test.copy()
            for name, transformer in pipeline.steps[:-1]:
                # サンプラーはtransformメソッドを持たないため、テスト時はスキップ
                if any(cls in str(type(transformer)) for cls in SAMPLING_CLASSES):
                    continue
                X_current = transformer.transform(X_current)
            self.X_test_transformed = X_current

        # 特徴量名（パイプライン変換後の正しい特徴量名を取得）
        from src.mlops.utils.pipeline_utils import get_pipeline_feature_names
        original_feature_names = (X_train.columns.tolist()
                                if hasattr(X_train, 'columns')
                                else [f'feature_{i}' for i in range(X_train.shape[1])])
        self.feature_names = get_pipeline_feature_names(self.pipeline, original_feature_names)

    @abstractmethod
    def create_plot(self, output_path: str) -> None:
        """可視化を作成してファイル保存"""
        pass


# ============================================================================
# YellowBrick可視化クラス群
# ============================================================================

class YellowBrickVisualizer(BaseVisualizer):
    """YellowBrick可視化の基底クラス"""

    def __init__(self, pipeline, X_train, y_train, X_test, y_test, task_type: str,
                 viz_class, target_names: Optional[List[str]] = None):
        super().__init__(pipeline, X_train, y_train, X_test, y_test, task_type)
        self.viz_class = viz_class
        self.target_names = target_names

    def create_plot(self, output_path: str) -> None:
        """YellowBrick可視化を作成"""
        # YellowBrick状態リセット
        plt.rcdefaults()
        plt.clf()
        plt.close('all')

        # matplotlib内部状態をクリア
        import matplotlib
        matplotlib.rcParams.clear()
        matplotlib.rcParams.update(matplotlib.rcParamsDefault)
        matplotlib.rcParams['font.family'] = 'DejaVu Sans'
        matplotlib.rcParams['font.sans-serif'] = ['DejaVu Sans', 'Ubuntu']

        # ValidationCurve専用処理
        if self.viz_class.__name__ == "ValidationCurve":
            # モデル種別に応じたパラメータを設定
            model_name = type(self.model).__name__
            if "LightGBM" in model_name or "LGBM" in model_name:
                param_name = "n_estimators"
                param_range = [10, 25, 50, 75, 100]
            elif "RandomForest" in model_name:
                param_name = "n_estimators"
                param_range = [10, 25, 50, 75, 100]
            elif "SVC" in model_name:
                param_name = "C"
                param_range = [0.1, 1, 10, 100]
            else:
                # デフォルト（汎用パラメータ）
                param_name = "n_estimators" if hasattr(self.model, "n_estimators") else "C"
                param_range = [10, 25, 50, 75, 100] if param_name == "n_estimators" else [0.1, 1, 10, 100]

            viz = self.viz_class(self.model, param_name=param_name, param_range=param_range)
        elif self.task_type == "classification" and self.target_names:
            viz = self.viz_class(self.model, classes=self.target_names)
        else:
            viz = self.viz_class(self.model)

        viz.fit(self.X_train_transformed, self.y_train)
        viz.score(self.X_test_transformed, self.y_test)
        viz.show(output_path)

        # YellowBrick後の完全クリーンアップ
        plt.rcdefaults()
        plt.clf()
        plt.close('all')
        viz.finalize()


class ClassBalanceVisualizer(BaseVisualizer):
    """ClassBalance専用可視化（特殊な処理が必要）"""

    def create_plot(self, output_path: str) -> None:
        from yellowbrick.target import ClassBalance

        # YellowBrick状態リセット
        plt.rcdefaults()
        plt.clf()
        plt.close('all')

        # matplotlib内部状態をクリア
        import matplotlib
        matplotlib.rcParams.clear()
        matplotlib.rcParams.update(matplotlib.rcParamsDefault)
        matplotlib.rcParams['font.family'] = 'DejaVu Sans'
        matplotlib.rcParams['font.sans-serif'] = ['DejaVu Sans', 'Ubuntu']

        viz = ClassBalance()
        # ClassBalanceは特殊で、ターゲットのみを使用
        viz.fit(self.y_train)
        viz.show(output_path)

        # YellowBrick後の完全クリーンアップ
        plt.rcdefaults()
        plt.clf()
        plt.close('all')
        viz.finalize()


# ============================================================================
# 解釈可視化クラス群
# ============================================================================

class PermutationImportanceVisualizer(BaseVisualizer):
    """Permutation Importance可視化"""

    def create_plot(self, output_path: str) -> None:
        from sklearn.inspection import permutation_importance
        from src.mlops.utils.pipeline_utils import get_transformed_data_with_feature_names

        # 可視化前にmatplotlibをクリア
        plt.clf()
        plt.close('all')

        # パイプライン変換後のテストデータと特徴量名を取得
        X_test_transformed, transformed_feature_names = get_transformed_data_with_feature_names(
            self.pipeline, self.X_test, self.X_test.columns.tolist()
        )

        scoring = 'accuracy' if self.task_type == "classification" else 'neg_mean_squared_error'
        perm_importance = permutation_importance(
            self.pipeline[-1], X_test_transformed, self.y_test,  # 最終段階のモデルのみ使用
            scoring=scoring, n_repeats=5, random_state=42
        )

        plt.figure(figsize=(10, 8))

        # 上位特徴量を選択
        max_features_to_show = min(15, len(perm_importance.importances_mean))
        indices = np.argsort(perm_importance.importances_mean)[-max_features_to_show:]

        plt.barh(range(len(indices)), perm_importance.importances_mean[indices])
        plt.yticks(range(len(indices)), [transformed_feature_names[i] for i in indices])
        plt.xlabel(f'Permutation Importance ({scoring})')
        plt.title(f'Permutation Feature Importance ({self.task_type.title()})')
        plt.tight_layout()
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        plt.close()


class SHAPBaseVisualizer(BaseVisualizer):
    """SHAP可視化の基底クラス"""

    def _create_explainer(self):
        """SHAP explainerを作成"""
        import shap
        if hasattr(self.model, 'predict_proba') and self.task_type == "classification":
            return shap.Explainer(self.model.predict_proba, self.X_train_transformed)
        else:
            return shap.Explainer(self.model.predict, self.X_train_transformed)

    def _get_shap_values(self, sample_size: int = 100):
        """SHAP値を計算"""
        explainer = self._create_explainer()
        sample_size = min(sample_size, len(self.X_test_transformed))
        X_sample = self.X_test_transformed[:sample_size]
        return explainer(X_sample), X_sample


class SHAPSummaryVisualizer(SHAPBaseVisualizer):
    """SHAP Summary Plot可視化"""

    def create_plot(self, output_path: str) -> None:
        import shap

        # 可視化前にmatplotlibをクリア
        plt.clf()
        plt.close('all')

        shap_values, X_sample = self._get_shap_values()

        plt.figure(figsize=(10, 8))
        if (self.task_type == "classification" and
            hasattr(shap_values, 'values') and
            len(shap_values.values.shape) == 3):
            # 多クラス分類
            shap.summary_plot(shap_values.values[:, :, 1], X_sample,
                            feature_names=self.feature_names, show=False)
        else:
            # 二値分類・回帰
            shap.summary_plot(shap_values.values, X_sample,
                            feature_names=self.feature_names, show=False)

        plt.title(f'SHAP Summary Plot ({self.task_type.title()})')
        plt.tight_layout()
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        plt.close()

        # 可視化後にも念のためクリア
        plt.clf()
        plt.close('all')


class SHAPDependenceVisualizer(SHAPBaseVisualizer):
    """SHAP Dependence Plot可視化"""

    def create_plot(self, output_path: str) -> None:
        import shap

        # 可視化前にmatplotlibをクリア
        plt.clf()
        plt.close('all')

        shap_values, X_sample = self._get_shap_values()

        if not hasattr(shap_values, 'values'):
            return

        # 最重要特徴量を特定
        if len(shap_values.values.shape) == 3:
            mean_abs_shap = np.mean(np.abs(shap_values.values[:, :, 1]), axis=0)
            shap_vals = shap_values.values[:, :, 1]
        else:
            mean_abs_shap = np.mean(np.abs(shap_values.values), axis=0)
            shap_vals = shap_values.values

        most_important_feature = np.argmax(mean_abs_shap)

        plt.figure(figsize=(10, 6))
        shap.dependence_plot(most_important_feature, shap_vals, X_sample,
                           feature_names=self.feature_names, show=False)
        plt.title(f'SHAP Dependence Plot ({self.task_type.title()})')
        plt.tight_layout()
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        plt.close()

        # 可視化後にも念のためクリア
        plt.clf()
        plt.close('all')


# ============================================================================
# ファクトリークラス
# ============================================================================

class VisualizationFactory:
    """可視化オブジェクトのファクトリー"""

    @staticmethod
    def create_visualizer(plot_type: str, pipeline, X_train, y_train, X_test, y_test,
                         task_type: str, target_names: Optional[List[str]] = None) -> BaseVisualizer:
        """可視化オブジェクトを作成"""

        if plot_type == "permutation_importance":
            return PermutationImportanceVisualizer(pipeline, X_train, y_train, X_test, y_test, task_type)

        elif plot_type == "shap_summary":
            return SHAPSummaryVisualizer(pipeline, X_train, y_train, X_test, y_test, task_type)

        elif plot_type == "shap_dependence":
            return SHAPDependenceVisualizer(pipeline, X_train, y_train, X_test, y_test, task_type)

        elif plot_type == "class_balance":
            return ClassBalanceVisualizer(pipeline, X_train, y_train, X_test, y_test, task_type)

        # YellowBrick可視化
        elif task_type == "classification":
            viz_mapping = {
                # 基本的な分類可視化
                "classification_report": "yellowbrick.classifier.ClassificationReport",
                "confusion_matrix": "yellowbrick.classifier.ConfusionMatrix",
                "roc_curve": "yellowbrick.classifier.ROCAUC",

                # 精度・再現率関連
                "precision_recall_curve": "yellowbrick.classifier.PrecisionRecallCurve",
                "class_prediction_error": "yellowbrick.classifier.ClassPredictionError",
                "discrimination_threshold": "yellowbrick.classifier.DiscriminationThreshold",  # 二値分類のみ

                # 閾値関連可視化
                "threshold_plot": "yellowbrick.classifier.ThresholdPlot",

                # モデル選択・評価
                "learning_curve": "yellowbrick.model_selection.LearningCurve",
                "validation_curve": "yellowbrick.model_selection.ValidationCurve",
                "feature_importances": "yellowbrick.model_selection.FeatureImportances",

                # クラスタリング・次元削減（分類でも使用可能）
                "silhouette_plot": "yellowbrick.cluster.SilhouetteVisualizer",
                "class_balance": "yellowbrick.target.ClassBalance"
            }
            if plot_type in viz_mapping:
                viz_class = _import_class(viz_mapping[plot_type])
                return YellowBrickVisualizer(pipeline, X_train, y_train, X_test, y_test,
                                           task_type, viz_class, target_names)

        else:  # regression
            viz_mapping = {
                # 基本的な回帰可視化
                "residuals_plot": "yellowbrick.regressor.ResidualsPlot",
                "prediction_error": "yellowbrick.regressor.PredictionError",

                # 詳細な回帰分析
                "cook_distance": "yellowbrick.regressor.CooksDistance",
                "alpha_selection": "yellowbrick.regressor.AlphaSelection",

                # モデル選択・評価
                "learning_curve": "yellowbrick.model_selection.LearningCurve",
                "validation_curve": "yellowbrick.model_selection.ValidationCurve",
                "feature_importances": "yellowbrick.model_selection.FeatureImportances"
            }
            if plot_type in viz_mapping:
                viz_class = _import_class(viz_mapping[plot_type])
                return YellowBrickVisualizer(pipeline, X_train, y_train, X_test, y_test,
                                           task_type, viz_class)

        raise ValueError(f"未対応の可視化タイプ: {plot_type}")


def _import_class(class_path: str):
    """クラスパスから動的インポート"""
    module_path, class_name = class_path.rsplit('.', 1)
    module = __import__(module_path, fromlist=[class_name])
    return getattr(module, class_name)


# ============================================================================
# メイン関数
# ============================================================================

def create_visualizations(pipeline, X_train, y_train, X_test, y_test, target_names_str, plot_types, cfg, task_type):
    """可視化を設定駆動で作成"""

    # YellowBrick状態リセットのためにmatplotlib完全初期化
    plt.rcdefaults()
    plt.clf()
    plt.close('all')
    import matplotlib
    matplotlib.rcParams.clear()
    matplotlib.rcParams.update(matplotlib.rcParamsDefault)
    # フォント警告抑制とフォント設定を再設定
    matplotlib.rcParams['font.family'] = 'DejaVu Sans'
    matplotlib.rcParams['font.sans-serif'] = ['DejaVu Sans', 'Ubuntu']

    with tempfile.TemporaryDirectory() as viz_dir:
        for plot_type in plot_types:
            try:
                visualizer = VisualizationFactory.create_visualizer(
                    plot_type, pipeline, X_train, y_train, X_test, y_test,
                    task_type, target_names_str
                )
                output_path = os.path.join(viz_dir, f"{plot_type}.png")
                visualizer.create_plot(output_path)

            except Exception as e:
                print(f"⚠️ 可視化エラー {plot_type}: {e}")
                # エラー時のクリーンアップ
                plt.clf()
                plt.close('all')

        # MLflowに保存
        for file in os.listdir(viz_dir):
            if file.endswith('.png'):
                mlflow.log_artifact(os.path.join(viz_dir, file), cfg.mlflow.artifacts.visualization_dir)

