"""
MLOps用データローダー

Hydra設定に基づいたCSVデータの読み込みと
基本情報の表示を提供する。
"""

from typing import Tuple, List, Any
import pandas as pd
from omegaconf import DictConfig


def load_csv_data(cfg: DictConfig) -> Tuple[pd.DataFrame, List[str], Any]:
    """
    Hydra設定に基づいてCSVデータを読み込む。

    Args:
        cfg: Hydraの設定オブジェクト
            data設定に以下を含む:
            - file_path: CSVファイルパス
            - target_column: ターゲット列名
            - read_csv_args: pandas.read_csvの引数（オプション）

    Returns:
        (DataFrame, 特徴量列名リスト, ターゲットのユニーク値)のタプル

    Note:
        'target_name'列が存在する場合は特徴量から除外される。
    """
    read_args = cfg.data.read_csv_args if hasattr(cfg.data, 'read_csv_args') else {}
    df = pd.read_csv(cfg.data.file_path, **read_args)

    target_col = cfg.data.target_column
    exclude_cols = [target_col, 'target_name'] if 'target_name' in df.columns else [target_col]
    feature_cols = [col for col in df.columns if col not in exclude_cols]
    target_names = df[target_col].unique() if target_col in df.columns else None

    print(f"📊 CSV: {cfg.data.file_path} - Shape: {df.shape} - Features: {len(feature_cols)} - Classes: {len(target_names) if target_names is not None else 'Unknown'}")
    return df, feature_cols, target_names