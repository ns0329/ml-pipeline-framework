"""データ処理・タスク検出ユーティリティ"""
import pandas as pd
from omegaconf import DictConfig


def get_dataset_name(file_path: str) -> str:
    """CSVファイルパスからデータセット名を取得"""
    return file_path.split('/')[-1].replace('.csv', '')


def detect_task_type(y):
    """ターゲット変数からタスクタイプを自動判定"""
    if pd.api.types.is_numeric_dtype(y):
        unique_values = len(y.unique())
        total_values = len(y)
        # ユニーク値が全体の5%以下かつ20個以下なら分類、それ以外は回帰
        if unique_values <= max(20, total_values * 0.05):
            return "classification"
        else:
            return "regression"
    else:
        return "classification"


def load_csv_data(cfg: DictConfig):
    """CSV データ読み込み関数"""
    # pandas.read_csv()引数の展開
    read_args = cfg.data.read_csv_args if hasattr(cfg.data, 'read_csv_args') else {}
    df = pd.read_csv(cfg.data.file_path, **read_args)

    target_col = cfg.data.target_column
    # target列とtarget_name列を除外して特徴量列を取得
    exclude_cols = [target_col, 'target_name'] if 'target_name' in df.columns else [target_col]
    feature_cols = [col for col in df.columns if col not in exclude_cols]
    target_names = df[target_col].unique() if target_col in df.columns else None

    print(f"📊 CSV: {cfg.data.file_path} - Shape: {df.shape} - Features: {len(feature_cols)} - Classes: {len(target_names) if target_names is not None else 'Unknown'}")
    return df, feature_cols, target_names