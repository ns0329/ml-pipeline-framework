"""データ型最適化テスト"""
import numpy as np
import pandas as pd
from src.mlops.features.dtype_optimizer import DtypeOptimizer

# テストデータ
X = pd.DataFrame({
    'feature1': [1.0, 2.0, 3.0],
    'feature2': [4.0, 5.0, 6.0]
})

print("=== データ型最適化テスト ===")
print(f"元データ型: {X.dtypes.iloc[0]}")

# LightGBM → float32
optimizer_lgb = DtypeOptimizer(model_name="lightgbm")
X_lgb = optimizer_lgb.fit_transform(X)
print(f"LightGBM用: {X_lgb.dtypes.iloc[0]}")

# LinearRegression → float64
optimizer_lr = DtypeOptimizer(model_name="LinearRegression")
X_lr = optimizer_lr.fit_transform(X)
print(f"LinearRegression用: {X_lr.dtypes.iloc[0]}")

# 強制指定
optimizer_force = DtypeOptimizer(force_dtype=np.float32)
X_force = optimizer_force.fit_transform(X)
print(f"強制float32: {X_force.dtypes.iloc[0]}")

print("✅ データ型最適化テスト完了")