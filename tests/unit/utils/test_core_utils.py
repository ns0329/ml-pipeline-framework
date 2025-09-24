"""core_utils.py の単体テスト"""

import pytest
import pandas as pd
import numpy as np
from unittest.mock import Mock, patch, MagicMock
from sklearn.feature_selection import f_classif, f_regression

from src.utils.core_utils import (
    import_class,
    resolve_function_references,
    get_dataset_name,
    detect_task_type
)


class TestImportClass:
    """import_class 関数の単体テスト"""

    def test_successful_import(self):
        """成功的なクラスインポートテスト"""
        # 実際のクラスをインポート
        DataFrame = import_class("pandas", "DataFrame")
        assert DataFrame == pd.DataFrame

        Series = import_class("pandas", "Series")
        assert Series == pd.Series

    def test_successful_import_nested_module(self):
        """ネストしたモジュールからのクラスインポートテスト"""
        StratifiedKFold = import_class("sklearn.model_selection", "StratifiedKFold")
        from sklearn.model_selection import StratifiedKFold as ExpectedClass
        assert StratifiedKFold == ExpectedClass

    def test_import_function(self):
        """関数のインポートテスト"""
        f_classif_func = import_class("sklearn.feature_selection", "f_classif")
        assert f_classif_func == f_classif

    def test_import_nonexistent_module(self):
        """存在しないモジュールのインポートエラーテスト"""
        with pytest.raises(ImportError) as excinfo:
            import_class("nonexistent.module", "SomeClass")

        assert "Failed to import SomeClass from nonexistent.module" in str(excinfo.value)

    def test_import_nonexistent_class(self):
        """存在しないクラスのインポートエラーテスト"""
        with pytest.raises(ImportError) as excinfo:
            import_class("pandas", "NonexistentClass")

        assert "Failed to import NonexistentClass from pandas" in str(excinfo.value)

    def test_import_with_special_characters(self):
        """特殊文字を含むモジュール名のテスト"""
        with pytest.raises(ImportError):
            import_class("invalid-module-name", "SomeClass")

    def test_empty_module_or_class_name(self):
        """空のモジュール名・クラス名のテスト"""
        with pytest.raises(ImportError):
            import_class("", "SomeClass")

        with pytest.raises(ImportError):
            import_class("pandas", "")


class TestResolveFunctionReferences:
    """resolve_function_references 関数の単体テスト"""

    def test_resolve_single_function_reference(self):
        """単一関数参照の解決テスト"""
        params = {
            "score_func": "sklearn.feature_selection.f_classif",
            "regular_param": "normal_value"
        }

        resolved = resolve_function_references(params)

        # f_classifが実際の関数オブジェクトに解決されることを確認
        assert resolved["score_func"] == f_classif
        assert resolved["regular_param"] == "normal_value"

    def test_resolve_multiple_function_references(self):
        """複数関数参照の解決テスト"""
        params = {
            "score_func": "sklearn.feature_selection.f_classif",
            "regression_func": "sklearn.feature_selection.f_regression",
            "normal_param": 42,
            "string_param": "test"
        }

        resolved = resolve_function_references(params)

        assert resolved["score_func"] == f_classif
        assert resolved["regression_func"] == f_regression
        assert resolved["normal_param"] == 42
        assert resolved["string_param"] == "test"

    def test_resolve_invalid_function_reference(self):
        """無効な関数参照のテスト"""
        params = {
            "invalid_func": "nonexistent.module.function",
            "another_invalid": "sklearn.nonexistent.function",
            "normal_param": "value"
        }

        resolved = resolve_function_references(params)

        # 無効な参照は文字列のまま残ることを確認
        assert resolved["invalid_func"] == "nonexistent.module.function"
        assert resolved["another_invalid"] == "sklearn.nonexistent.function"
        assert resolved["normal_param"] == "value"

    def test_resolve_non_function_strings(self):
        """関数参照でない文字列のテスト"""
        params = {
            "url": "http://example.com",
            "file_path": "/path/to/file.txt",
            "relative_path": "./data/file.csv",
            "simple_string": "no_dots_here",
            "function_ref": "sklearn.feature_selection.f_classif"
        }

        resolved = resolve_function_references(params)

        # URL、ファイルパス、単純文字列は変更されない
        assert resolved["url"] == "http://example.com"
        assert resolved["file_path"] == "/path/to/file.txt"
        assert resolved["relative_path"] == "./data/file.csv"
        assert resolved["simple_string"] == "no_dots_here"

        # 関数参照のみ解決される
        assert resolved["function_ref"] == f_classif

    def test_resolve_nested_dict_handling(self):
        """ネストした辞書の処理テスト（現在の実装では対象外）"""
        params = {
            "outer_param": "sklearn.feature_selection.f_classif",
            "nested": {
                "inner_func": "sklearn.feature_selection.f_regression"
            }
        }

        resolved = resolve_function_references(params)

        # 現在の実装では最上位レベルのみ処理
        assert resolved["outer_param"] == f_classif
        assert resolved["nested"]["inner_func"] == "sklearn.feature_selection.f_regression"

    def test_resolve_empty_params(self):
        """空のパラメータ辞書のテスト"""
        params = {}
        resolved = resolve_function_references(params)
        assert resolved == {}

    def test_resolve_non_string_values(self):
        """非文字列値のテスト"""
        params = {
            "int_param": 42,
            "float_param": 3.14,
            "bool_param": True,
            "none_param": None,
            "list_param": [1, 2, 3],
            "function_ref": "sklearn.feature_selection.f_classif"
        }

        resolved = resolve_function_references(params)

        # 非文字列値は変更されない
        assert resolved["int_param"] == 42
        assert resolved["float_param"] == 3.14
        assert resolved["bool_param"] == True
        assert resolved["none_param"] is None
        assert resolved["list_param"] == [1, 2, 3]

        # 文字列の関数参照のみ解決
        assert resolved["function_ref"] == f_classif


class TestGetDatasetName:
    """get_dataset_name 関数の単体テスト"""

    def test_simple_csv_filename(self):
        """シンプルなCSVファイル名のテスト"""
        assert get_dataset_name("iris.csv") == "iris"
        assert get_dataset_name("wine.csv") == "wine"

    def test_full_path_csv_filename(self):
        """フルパスCSVファイル名のテスト"""
        assert get_dataset_name("/data/raw/iris.csv") == "iris"
        assert get_dataset_name("/home/user/datasets/titanic.csv") == "titanic"
        assert get_dataset_name("data/processed/features.csv") == "features"

    def test_deep_nested_path(self):
        """深くネストしたパスのテスト"""
        path = "/very/deep/nested/path/to/data/boston_housing.csv"
        assert get_dataset_name(path) == "boston_housing"

    def test_filename_with_underscores(self):
        """アンダースコア含むファイル名のテスト"""
        assert get_dataset_name("breast_cancer.csv") == "breast_cancer"
        assert get_dataset_name("/data/california_housing.csv") == "california_housing"

    def test_filename_with_hyphens(self):
        """ハイフン含むファイル名のテスト"""
        assert get_dataset_name("e-commerce-data.csv") == "e-commerce-data"
        assert get_dataset_name("/path/multi-class-classification.csv") == "multi-class-classification"

    def test_non_csv_extension(self):
        """非CSV拡張子のテスト（現在の実装の動作確認）"""
        # .csvを単純に置換するので、.csvが含まれないファイルは変更されない
        assert get_dataset_name("data.txt") == "data.txt"
        assert get_dataset_name("model.pkl") == "model.pkl"

    def test_multiple_csv_in_filename(self):
        """ファイル名に複数回.csvが含まれる場合のテスト"""
        # 最初の.csvのみが置換される（Python strのreplace動作）
        assert get_dataset_name("csv_data.csv") == "_data"

    def test_empty_filename(self):
        """空のファイル名のテスト"""
        assert get_dataset_name("") == ""

    def test_only_csv_extension(self):
        """拡張子のみのテスト"""
        assert get_dataset_name(".csv") == "."
        assert get_dataset_name("/.csv") == "."

    def test_windows_path_separator(self):
        """Windows形式のパス区切り文字のテスト"""
        # 現在の実装では'/'のみをサポート
        assert get_dataset_name("C:\\data\\iris.csv") == "C:\\data\\iris"


class TestDetectTaskType:
    """detect_task_type 関数の単体テスト"""

    def test_binary_classification(self):
        """二値分類の判定テスト"""
        y_binary = pd.Series([0, 1, 1, 0, 1, 0, 0, 1])
        assert detect_task_type(y_binary) == "classification"

        y_binary_str = pd.Series(["yes", "no", "yes", "no", "yes"])
        assert detect_task_type(y_binary_str) == "classification"

    def test_multiclass_classification(self):
        """多クラス分類の判定テスト"""
        y_multiclass = pd.Series([0, 1, 2, 0, 1, 2, 1, 0])
        assert detect_task_type(y_multiclass) == "classification"

        y_multiclass_str = pd.Series(["cat", "dog", "bird", "cat", "dog"])
        assert detect_task_type(y_multiclass_str) == "classification"

    def test_regression_continuous(self):
        """連続値回帰の判定テスト"""
        y_continuous = pd.Series(np.random.randn(100))
        assert detect_task_type(y_continuous) == "regression"

        y_price = pd.Series([1.5, 2.3, 5.7, 8.9, 12.1, 15.6, 20.3, 25.8])
        assert detect_task_type(y_price) == "regression"

    def test_regression_many_unique_values(self):
        """多くのユニーク値を持つ回帰の判定テスト"""
        # 100個のユニークな値（全体の5%を超える）
        y_many_unique = pd.Series(range(100))
        assert detect_task_type(y_many_unique) == "regression"

    def test_borderline_case_small_dataset(self):
        """小さなデータセットの境界ケーステスト"""
        # 20個以下のユニーク値でも、全体の5%以下なら分類
        y_small = pd.Series([1, 2, 3, 4, 5] * 50)  # 250個のデータ、5個のユニーク値（2%）
        assert detect_task_type(y_small) == "classification"

    def test_borderline_case_exactly_20_unique(self):
        """ちょうど20個のユニーク値のテスト"""
        y_20_unique = pd.Series(list(range(20)) * 10)  # 200個のデータ、20個のユニーク値（10%）
        assert detect_task_type(y_20_unique) == "classification"

    def test_borderline_case_21_unique(self):
        """21個のユニーク値のテスト（回帰判定される）"""
        y_21_unique = pd.Series(list(range(21)) * 10)  # 210個のデータ、21個のユニーク値（10%）
        assert detect_task_type(y_21_unique) == "regression"

    def test_exactly_5_percent_threshold(self):
        """ちょうど5%の閾値テスト"""
        # 1000個のデータ、50個のユニーク値（ちょうど5%）
        y_5_percent = pd.Series(list(range(50)) * 20)
        assert detect_task_type(y_5_percent) == "classification"

        # 1000個のデータ、51個のユニーク値（5.1%）
        y_over_5_percent = pd.Series(list(range(51)) + [0] * 949)
        assert detect_task_type(y_over_5_percent) == "regression"

    def test_non_numeric_data(self):
        """非数値データのテスト"""
        y_text = pd.Series(["spam", "ham", "spam", "ham"])
        assert detect_task_type(y_text) == "classification"

        y_categories = pd.Series(["A", "B", "C", "A", "B"])
        assert detect_task_type(y_categories) == "classification"

    def test_mixed_numeric_string(self):
        """数値文字列混在データのテスト"""
        y_mixed = pd.Series(["1", "2", "3", "1", "2"])
        assert detect_task_type(y_mixed) == "classification"

    def test_numpy_array_input(self):
        """numpy array入力のテスト"""
        y_array_classification = np.array([0, 1, 1, 0, 1])
        assert detect_task_type(y_array_classification) == "classification"

        y_array_regression = np.random.randn(100)
        assert detect_task_type(y_array_regression) == "regression"

    def test_single_value(self):
        """単一値のテスト"""
        y_single = pd.Series([42])
        assert detect_task_type(y_single) == "classification"  # 1個のユニーク値

    def test_empty_series(self):
        """空のSeriesのテスト"""
        y_empty = pd.Series([], dtype=float)
        assert detect_task_type(y_empty) == "classification"  # 0個のユニーク値

    def test_series_with_nan(self):
        """NaN値を含むSeriesのテスト"""
        y_with_nan = pd.Series([1, 2, np.nan, 1, 2, np.nan])
        # NaNもユニークな値として扱われるかを確認
        result = detect_task_type(y_with_nan)
        assert result in ["classification", "regression"]  # 実装依存

    def test_float_integers(self):
        """浮動小数点だが整数値のテスト"""
        y_float_int = pd.Series([1.0, 2.0, 3.0, 1.0, 2.0])
        assert detect_task_type(y_float_int) == "classification"

    def test_large_dataset_few_classes(self):
        """大きなデータセットで少ないクラス数のテスト"""
        # 10000個のデータ、3個のクラス（0.03% << 5%）
        y_large_few_classes = pd.Series([0, 1, 2] * 3334)
        assert detect_task_type(y_large_few_classes) == "classification"

    def test_boolean_data(self):
        """ブール値データのテスト"""
        y_bool = pd.Series([True, False, True, False, True])
        assert detect_task_type(y_bool) == "classification"