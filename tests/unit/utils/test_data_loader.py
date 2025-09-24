"""data_loader.py の単体テスト"""

import pytest
import pandas as pd
import numpy as np
import tempfile
import os
from unittest.mock import Mock, patch, mock_open
from omegaconf import DictConfig

from src.mlops.utils.data_loader import load_csv_data


class TestLoadCsvData:
    """load_csv_data 関数の単体テスト"""

    @patch('src.mlops.utils.data_loader.pd.read_csv')
    def test_basic_csv_loading(self, mock_read_csv):
        """基本的なCSV読み込みテスト"""
        # Mock DataFrame作成
        mock_df = pd.DataFrame({
            'feature_0': [1, 2, 3, 4, 5],
            'feature_1': [6, 7, 8, 9, 10],
            'target': [0, 1, 0, 1, 0]
        })
        mock_read_csv.return_value = mock_df

        cfg = DictConfig({
            "data": {
                "file_path": "/path/to/data.csv",
                "target_column": "target"
            }
        })

        with patch('builtins.print'):
            df, feature_cols, target_names = load_csv_data(cfg)

        # アサーション
        mock_read_csv.assert_called_once_with("/path/to/data.csv")
        pd.testing.assert_frame_equal(df, mock_df)
        assert feature_cols == ['feature_0', 'feature_1']
        np.testing.assert_array_equal(target_names, [0, 1])

    @patch('src.mlops.utils.data_loader.pd.read_csv')
    def test_csv_loading_with_read_csv_args(self, mock_read_csv):
        """read_csv引数付きCSV読み込みテスト"""
        # Mock DataFrame作成
        mock_df = pd.DataFrame({
            'col1': ['a', 'b', 'c'],
            'col2': [1.1, 2.2, 3.3],
            'label': ['class_A', 'class_B', 'class_A']
        })
        mock_read_csv.return_value = mock_df

        cfg = DictConfig({
            "data": {
                "file_path": "/path/to/data.csv",
                "target_column": "label",
                "read_csv_args": {
                    "sep": ";",
                    "encoding": "utf-8",
                    "na_values": ["N/A", "null"]
                }
            }
        })

        with patch('builtins.print'):
            df, feature_cols, target_names = load_csv_data(cfg)

        # read_csv_argsが正しく渡されることを確認
        mock_read_csv.assert_called_once_with(
            "/path/to/data.csv",
            sep=";",
            encoding="utf-8",
            na_values=["N/A", "null"]
        )
        pd.testing.assert_frame_equal(df, mock_df)
        assert feature_cols == ['col1', 'col2']
        np.testing.assert_array_equal(target_names, ['class_A', 'class_B'])

    @patch('src.mlops.utils.data_loader.pd.read_csv')
    def test_csv_loading_with_target_name_column(self, mock_read_csv):
        """target_name列含むCSV読み込みテスト"""
        # Mock DataFrame作成（target_name列を含む）
        mock_df = pd.DataFrame({
            'feature_0': [1, 2, 3],
            'feature_1': [4, 5, 6],
            'target': [0, 1, 0],
            'target_name': ['negative', 'positive', 'negative']
        })
        mock_read_csv.return_value = mock_df

        cfg = DictConfig({
            "data": {
                "file_path": "/data/labeled_data.csv",
                "target_column": "target"
            }
        })

        with patch('builtins.print'):
            df, feature_cols, target_names = load_csv_data(cfg)

        # target_name列が特徴量から除外されることを確認
        assert feature_cols == ['feature_0', 'feature_1']
        assert 'target' not in feature_cols
        assert 'target_name' not in feature_cols
        np.testing.assert_array_equal(target_names, [0, 1])

    @patch('src.mlops.utils.data_loader.pd.read_csv')
    def test_csv_loading_without_target_column(self, mock_read_csv):
        """ターゲット列が存在しない場合のテスト"""
        # Mock DataFrame作成（ターゲット列なし）
        mock_df = pd.DataFrame({
            'feature_0': [1, 2, 3],
            'feature_1': [4, 5, 6],
            'feature_2': [7, 8, 9]
        })
        mock_read_csv.return_value = mock_df

        cfg = DictConfig({
            "data": {
                "file_path": "/data/no_target.csv",
                "target_column": "nonexistent_target"
            }
        })

        with patch('builtins.print'):
            df, feature_cols, target_names = load_csv_data(cfg)

        # 全ての列が特徴量として扱われることを確認
        assert feature_cols == ['feature_0', 'feature_1', 'feature_2']
        assert target_names is None

    @patch('src.mlops.utils.data_loader.pd.read_csv')
    def test_csv_loading_without_read_csv_args(self, mock_read_csv):
        """read_csv_args設定がない場合のテスト"""
        # Mock DataFrame作成
        mock_df = pd.DataFrame({
            'x': [1, 2],
            'y': [3, 4],
            'z': [0, 1]
        })
        mock_read_csv.return_value = mock_df

        cfg = DictConfig({
            "data": {
                "file_path": "/simple/data.csv",
                "target_column": "z"
                # read_csv_args設定なし
            }
        })

        with patch('builtins.print'):
            df, feature_cols, target_names = load_csv_data(cfg)

        # デフォルトのread_csvが呼ばれることを確認
        mock_read_csv.assert_called_once_with("/simple/data.csv")
        assert feature_cols == ['x', 'y']
        np.testing.assert_array_equal(target_names, [0, 1])

    @patch('src.mlops.utils.data_loader.pd.read_csv')
    def test_multiclass_target_handling(self, mock_read_csv):
        """多クラスターゲットの処理テスト"""
        # Mock DataFrame作成
        mock_df = pd.DataFrame({
            'feature_a': [1, 2, 3, 4, 5, 6],
            'feature_b': [7, 8, 9, 10, 11, 12],
            'class': ['A', 'B', 'C', 'A', 'B', 'C']
        })
        mock_read_csv.return_value = mock_df

        cfg = DictConfig({
            "data": {
                "file_path": "/multiclass/data.csv",
                "target_column": "class"
            }
        })

        with patch('builtins.print'):
            df, feature_cols, target_names = load_csv_data(cfg)

        assert feature_cols == ['feature_a', 'feature_b']
        # ユニークな値の順序は保証されないので、setで比較
        assert set(target_names) == {'A', 'B', 'C'}
        assert len(target_names) == 3

    @patch('src.mlops.utils.data_loader.pd.read_csv')
    def test_numeric_target_handling(self, mock_read_csv):
        """数値ターゲットの処理テスト"""
        # Mock DataFrame作成
        mock_df = pd.DataFrame({
            'feature1': [1.1, 2.2, 3.3],
            'feature2': [4.4, 5.5, 6.6],
            'price': [100.5, 200.7, 150.3]
        })
        mock_read_csv.return_value = mock_df

        cfg = DictConfig({
            "data": {
                "file_path": "/regression/data.csv",
                "target_column": "price"
            }
        })

        with patch('builtins.print'):
            df, feature_cols, target_names = load_csv_data(cfg)

        assert feature_cols == ['feature1', 'feature2']
        np.testing.assert_array_equal(target_names, [100.5, 150.3, 200.7])

    @patch('src.mlops.utils.data_loader.pd.read_csv')
    def test_print_output_information(self, mock_read_csv):
        """情報出力テスト"""
        # Mock DataFrame作成
        mock_df = pd.DataFrame({
            'f1': range(100),
            'f2': range(100, 200),
            'f3': range(200, 300),
            'target': [0, 1] * 50
        })
        mock_read_csv.return_value = mock_df

        cfg = DictConfig({
            "data": {
                "file_path": "/large/dataset.csv",
                "target_column": "target"
            }
        })

        with patch('builtins.print') as mock_print:
            df, feature_cols, target_names = load_csv_data(cfg)

        # print文が呼ばれることを確認
        mock_print.assert_called_once()
        print_message = mock_print.call_args[0][0]

        assert "/large/dataset.csv" in print_message
        assert "Shape: (100, 4)" in print_message
        assert "Features: 3" in print_message
        assert "Classes: 2" in print_message

    @patch('src.mlops.utils.data_loader.pd.read_csv')
    def test_empty_dataframe_handling(self, mock_read_csv):
        """空のDataFrameの処理テスト"""
        # 空のMock DataFrame作成
        mock_df = pd.DataFrame()
        mock_read_csv.return_value = mock_df

        cfg = DictConfig({
            "data": {
                "file_path": "/empty/data.csv",
                "target_column": "target"
            }
        })

        with patch('builtins.print'):
            df, feature_cols, target_names = load_csv_data(cfg)

        assert df.empty
        assert feature_cols == []
        assert target_names is None

    @patch('src.mlops.utils.data_loader.pd.read_csv')
    def test_single_column_dataframe(self, mock_read_csv):
        """単一列DataFrameの処理テスト"""
        # 単一列のMock DataFrame作成
        mock_df = pd.DataFrame({'only_target': [1, 2, 3, 4, 5]})
        mock_read_csv.return_value = mock_df

        cfg = DictConfig({
            "data": {
                "file_path": "/single/column.csv",
                "target_column": "only_target"
            }
        })

        with patch('builtins.print'):
            df, feature_cols, target_names = load_csv_data(cfg)

        assert feature_cols == []  # ターゲット列のみなので特徴量は空
        np.testing.assert_array_equal(target_names, [1, 2, 3, 4, 5])

    @patch('src.mlops.utils.data_loader.pd.read_csv')
    def test_duplicate_target_values(self, mock_read_csv):
        """重複するターゲット値の処理テスト"""
        # 重複するターゲット値を含むMock DataFrame作成
        mock_df = pd.DataFrame({
            'feature': [1, 2, 3, 4, 5, 6],
            'target': [0, 0, 1, 1, 0, 1]
        })
        mock_read_csv.return_value = mock_df

        cfg = DictConfig({
            "data": {
                "file_path": "/duplicate/targets.csv",
                "target_column": "target"
            }
        })

        with patch('builtins.print'):
            df, feature_cols, target_names = load_csv_data(cfg)

        assert feature_cols == ['feature']
        # 重複は除去されてユニークな値のみ
        assert set(target_names) == {0, 1}
        assert len(target_names) == 2

    def test_file_not_found_error(self):
        """ファイルが見つからない場合のエラーテスト"""
        cfg = DictConfig({
            "data": {
                "file_path": "/nonexistent/file.csv",
                "target_column": "target"
            }
        })

        with pytest.raises(FileNotFoundError):
            load_csv_data(cfg)

    @patch('src.mlops.utils.data_loader.pd.read_csv')
    def test_invalid_read_csv_args(self, mock_read_csv):
        """無効なread_csv引数のテスト"""
        # read_csvでエラーが発生するように設定
        mock_read_csv.side_effect = ValueError("Invalid separator")

        cfg = DictConfig({
            "data": {
                "file_path": "/data/invalid.csv",
                "target_column": "target",
                "read_csv_args": {
                    "sep": "invalid_separator"
                }
            }
        })

        with pytest.raises(ValueError) as excinfo:
            load_csv_data(cfg)

        assert "Invalid separator" in str(excinfo.value)

    def test_missing_data_config(self):
        """data設定が存在しない場合のテスト"""
        cfg = DictConfig({
            "other_config": {"param": "value"}
            # data設定なし
        })

        with pytest.raises(AttributeError):
            load_csv_data(cfg)

    def test_missing_file_path_config(self):
        """file_path設定が存在しない場合のテスト"""
        cfg = DictConfig({
            "data": {
                "target_column": "target"
                # file_path設定なし
            }
        })

        with pytest.raises(AttributeError):
            load_csv_data(cfg)

    def test_missing_target_column_config(self):
        """target_column設定が存在しない場合のテスト"""
        cfg = DictConfig({
            "data": {
                "file_path": "/path/to/data.csv"
                # target_column設定なし
            }
        })

        with pytest.raises(AttributeError):
            load_csv_data(cfg)

    def test_real_csv_file_integration(self):
        """実際のCSVファイルでの統合テスト"""
        # 一時CSVファイルを作成
        with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as f:
            f.write("feature1,feature2,feature3,target\n")
            f.write("1.0,2.0,3.0,0\n")
            f.write("4.0,5.0,6.0,1\n")
            f.write("7.0,8.0,9.0,0\n")
            temp_file_path = f.name

        try:
            cfg = DictConfig({
                "data": {
                    "file_path": temp_file_path,
                    "target_column": "target"
                }
            })

            with patch('builtins.print'):
                df, feature_cols, target_names = load_csv_data(cfg)

            # 実際のファイル読み込みが正常に動作することを確認
            assert df.shape == (3, 4)
            assert feature_cols == ['feature1', 'feature2', 'feature3']
            assert set(target_names) == {0, 1}
            assert df['feature1'].tolist() == [1.0, 4.0, 7.0]
            assert df['target'].tolist() == [0, 1, 0]

        finally:
            # 一時ファイルをクリーンアップ
            os.unlink(temp_file_path)

    def test_complex_read_csv_args_integration(self):
        """複雑なread_csv引数での統合テスト"""
        # カスタム区切り文字とヘッダーを持つ一時CSVファイルを作成
        with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as f:
            f.write("# This is a comment\n")
            f.write("col_a;col_b;col_c;label\n")
            f.write("1.5;2.5;3.5;positive\n")
            f.write("4.5;5.5;6.5;negative\n")
            temp_file_path = f.name

        try:
            cfg = DictConfig({
                "data": {
                    "file_path": temp_file_path,
                    "target_column": "label",
                    "read_csv_args": {
                        "sep": ";",
                        "comment": "#",
                        "dtype": {"col_a": "float32", "col_b": "float32", "col_c": "float32"}
                    }
                }
            })

            with patch('builtins.print'):
                df, feature_cols, target_names = load_csv_data(cfg)

            # カスタム引数での読み込みが正常に動作することを確認
            assert df.shape == (2, 4)
            assert feature_cols == ['col_a', 'col_b', 'col_c']
            assert set(target_names) == {'positive', 'negative'}
            assert df['col_a'].dtype == 'float32'
            assert df['col_b'].dtype == 'float32'

        finally:
            # 一時ファイルをクリーンアップ
            os.unlink(temp_file_path)