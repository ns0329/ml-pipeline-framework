"""データ型最適化Transformer（超シンプル版）"""
import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin


class DtypeOptimizer(BaseEstimator, TransformerMixin):
    """モデル種別に応じた最適データ型変換"""

    def __init__(self, model_name="auto", force_dtype=None):
        self.model_name = model_name
        self.force_dtype = force_dtype
        self._dtype = None

    def fit(self, X, y=None):
        # シンプル判定ロジック
        if self.force_dtype:
            self._dtype = self.force_dtype
        else:
            precision_models = ["LinearRegression", "LogisticRegression", "SVC", "Ridge", "Lasso"]
            self._dtype = np.float64 if any(m in str(self.model_name) for m in precision_models) else np.float32
        return self

    def transform(self, X):
        print(f"🔄 データ型最適化: {self._dtype}")
        return X.astype(self._dtype)

    def get_feature_names_out(self, input_features=None):
        """変換後の特徴量名を返す（データ型変更のみなので特徴量名は変更されない）"""
        if input_features is None:
            raise ValueError("input_features must be provided")
        return np.array(input_features)