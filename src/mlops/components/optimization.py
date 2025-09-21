"""最適化コンポーネント（Optuna）"""
import optuna
from sklearn.model_selection import cross_val_score
from src.mlops.components.pipeline import create_pipeline
from src.utils.cv_utils import create_cv_strategy


class OptunaOptimizer:
    """Optuna最適化実行クラス"""

    def __init__(self, cfg, X_train, y_train, task_type):
        self.cfg = cfg
        self.X_train = X_train
        self.y_train = y_train
        self.task_type = task_type

    def objective(self, trial):
        """Optuna最適化の目的関数"""
        try:
            # パイプライン構築（試行用）
            pipeline = create_pipeline(self.cfg, trial=trial)

            # タスクタイプ別評価
            if self.task_type == "classification":
                scoring = self.cfg.optuna.scoring.classification
            else:  # regression
                scoring = self.cfg.optuna.scoring.regression

            # CV戦略を作成してクロスバリデーション実行
            cv_strategy = create_cv_strategy(self.cfg, self.X_train, self.y_train)
            cv_scores = cross_val_score(
                pipeline, self.X_train, self.y_train,
                cv=cv_strategy,
                scoring=scoring,
                n_jobs=-1
            )

            return cv_scores.mean()

        except Exception as e:
            print(f"⚠️ Trial failed: {e}")
            return float('-inf') if self.cfg.optuna.direction == "maximize" else float('inf')

    def optimize(self):
        """最適化実行"""
        print(f"🎯 Optuna最適化開始 | {self.cfg.optuna.n_trials} trials | {self.cfg.optuna.direction}")

        study = optuna.create_study(
            direction=self.cfg.optuna.direction,
            study_name=self.cfg.optuna.study_name
        )

        study.optimize(self.objective, n_trials=self.cfg.optuna.n_trials)

        print(f"🎯 Optuna最適化完了 | Best: {study.best_value:.3f} | Params: {study.best_params}")

        return study.best_params, study.best_value