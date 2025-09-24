"""
クロスバリデーション戦略ユーティリティ

MLOpsパイプライン用のクロスバリデーション戦略を
設定から動的に生成する。
"""

from omegaconf import DictConfig
from src.utils.core_utils import import_class


def create_cv_strategy(cfg: DictConfig):
    """
    設定に基づいてクロスバリデーション戦略を生成する。

    Args:
        cfg: Hydraの設定オブジェクト
            評価設定（evaluation.cv_strategy）に以下を含む:
            - module: モジュール名
            - class: クラス名
            - params: クラスのパラメータ

    Returns:
        生成されたクロスバリデーション戦略インスタンス

    Example:
        >>> cv = create_cv_strategy(cfg)
        >>> for train_idx, test_idx in cv.split(X, y):
        ...     # クロスバリデーション処理
    """
    cv_config = cfg.evaluation.cv_strategy
    cv_class = import_class(cv_config.module, cv_config["class"])
    return cv_class(**cv_config.params)