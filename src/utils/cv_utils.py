"""クロスバリデーション戦略のユーティリティ"""
from sklearn.model_selection import (
    KFold, StratifiedKFold, GroupKFold, TimeSeriesSplit,
    RepeatedKFold, RepeatedStratifiedKFold
)


def create_cv_strategy(cfg, X, y, groups=None):
    """configに基づいてCV戦略を作成"""
    strategy = cfg.evaluation.cv_strategy
    n_splits = cfg.evaluation.cv_folds

    # CV戦略のパラメータ（設定されている場合のみ使用）
    cv_params = getattr(cfg.evaluation, 'cv_params', {})

    if strategy == "KFold":
        return KFold(
            n_splits=n_splits,
            shuffle=cv_params.get('shuffle', True),
            random_state=cv_params.get('random_state', None)
        )

    elif strategy == "StratifiedKFold":
        return StratifiedKFold(
            n_splits=n_splits,
            shuffle=cv_params.get('shuffle', True),
            random_state=cv_params.get('random_state', None)
        )

    elif strategy == "GroupKFold":
        if groups is None:
            raise ValueError("GroupKFold requires groups parameter")
        return GroupKFold(n_splits=n_splits)

    elif strategy == "TimeSeriesSplit":
        return TimeSeriesSplit(
            n_splits=n_splits,
            max_train_size=cv_params.get('max_train_size', None),
            test_size=cv_params.get('test_size', None),
            gap=cv_params.get('gap', 0)
        )

    elif strategy == "RepeatedKFold":
        return RepeatedKFold(
            n_splits=n_splits,
            n_repeats=cv_params.get('n_repeats', 10),
            random_state=cv_params.get('random_state', None)
        )

    elif strategy == "RepeatedStratifiedKFold":
        return RepeatedStratifiedKFold(
            n_splits=n_splits,
            n_repeats=cv_params.get('n_repeats', 10),
            random_state=cv_params.get('random_state', None)
        )

    else:
        raise ValueError(f"Unsupported CV strategy: {strategy}")


def detect_best_cv_strategy(X, y, task_type):
    """データとタスクタイプに基づいて推奨CV戦略を提案"""
    if task_type == "classification":
        # 分類の場合はクラス分布を考慮
        return "StratifiedKFold"
    else:
        # 回帰の場合は通常のKFold
        return "KFold"