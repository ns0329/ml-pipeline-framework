"""クロスバリデーション戦略のユーティリティ"""
from src.utils.import_utils import import_class


def create_cv_strategy(cfg):
    """configに基づいてCV戦略を作成（パイプライン方式）"""
    cv_config = cfg.evaluation.cv_strategy
    cv_class = import_class(cv_config.module, cv_config["class"])
    return cv_class(**cv_config.params)