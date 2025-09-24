"""
汎用ユーティリティ関数群

このモジュールは、MLOps、EDA、データサイエンス探索など、
様々なコンテキストで使用可能な汎用的な関数を提供します。
"""

import importlib
import pandas as pd


def import_class(module_name: str, class_name: str):
    """
    動的にクラスをインポートする。

    Args:
        module_name: モジュール名（例: 'sklearn.model_selection'）
        class_name: クラス名（例: 'StratifiedKFold'）

    Returns:
        インポートされたクラスオブジェクト

    Raises:
        ImportError: クラスのインポートに失敗した場合
    """
    try:
        module = importlib.import_module(module_name)
        return getattr(module, class_name)
    except (ImportError, AttributeError) as e:
        raise ImportError(f"Failed to import {class_name} from {module_name}: {e}")


def resolve_function_references(params: dict) -> dict:
    """
    文字列形式の関数参照を実際の関数オブジェクトに解決する。

    'module.submodule.function'形式の文字列を検出し、
    対応する関数をインポートして置換する。

    Args:
        params: パラメータ辞書

    Returns:
        関数参照が解決された後のパラメータ辞書

    Example:
        >>> params = {'score_func': 'sklearn.feature_selection.f_classif'}
        >>> resolved = resolve_function_references(params)
        >>> # params['score_func']は実際の関数オブジェクトになる
    """
    for key, value in params.items():
        if isinstance(value, str) and '.' in value and not value.startswith(('http', 'file', '/', './')):
            try:
                module_path, func_name = value.rsplit('.', 1)
                module = importlib.import_module(module_path)
                params[key] = getattr(module, func_name)
            except (ImportError, AttributeError):
                pass  # インポート失敗時は文字列のまま保持
    return params


def get_dataset_name(file_path: str) -> str:
    """
    ファイルパスからデータセット名を抽出する。

    Args:
        file_path: CSVファイルのパス

    Returns:
        拡張子を除いたファイル名

    Example:
        >>> get_dataset_name('/data/raw/iris.csv')
        'iris'
    """
    return file_path.split('/')[-1].replace('.csv', '')


def detect_task_type(y) -> str:
    """
    ターゲット変数からタスクタイプ（分類/回帰）を自動判定する。

    判定基準:
    - ユニーク値が全体の5%以下、かつ20個以下の場合: 分類
    - それ以外: 回帰
    - 非数値データ: 分類

    Args:
        y: ターゲット変数（pandas Series or array-like）

    Returns:
        'classification' または 'regression'
    """
    if pd.api.types.is_numeric_dtype(y):
        unique_values = len(pd.Series(y).unique())
        total_values = len(y)

        if unique_values <= max(20, total_values * 0.05):
            return "classification"
        else:
            return "regression"
    else:
        return "classification"