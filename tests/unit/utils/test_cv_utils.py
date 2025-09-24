"""cv_utils.py の単体テスト"""

import pytest
from unittest.mock import Mock, patch
from omegaconf import DictConfig

from src.mlops.utils.cv_utils import create_cv_strategy


class TestCreateCvStrategy:
    """create_cv_strategy 関数の単体テスト"""

    @patch('src.mlops.utils.cv_utils.import_class')
    def test_stratified_kfold_creation(self, mock_import_class):
        """StratifiedKFold作成テスト"""
        # Mock設定
        mock_cv_class = Mock()
        mock_cv_instance = Mock()
        mock_cv_class.return_value = mock_cv_instance
        mock_import_class.return_value = mock_cv_class

        cfg = DictConfig({
            "evaluation": {
                "cv_strategy": {
                    "module": "sklearn.model_selection",
                    "class": "StratifiedKFold",
                    "params": {
                        "n_splits": 5,
                        "shuffle": True,
                        "random_state": 42
                    }
                }
            }
        })

        result = create_cv_strategy(cfg)

        # アサーション
        mock_import_class.assert_called_once_with("sklearn.model_selection", "StratifiedKFold")
        mock_cv_class.assert_called_once_with(n_splits=5, shuffle=True, random_state=42)
        assert result == mock_cv_instance

    @patch('src.mlops.utils.cv_utils.import_class')
    def test_kfold_creation(self, mock_import_class):
        """KFold作成テスト"""
        # Mock設定
        mock_cv_class = Mock()
        mock_cv_instance = Mock()
        mock_cv_class.return_value = mock_cv_instance
        mock_import_class.return_value = mock_cv_class

        cfg = DictConfig({
            "evaluation": {
                "cv_strategy": {
                    "module": "sklearn.model_selection",
                    "class": "KFold",
                    "params": {
                        "n_splits": 3,
                        "shuffle": False
                    }
                }
            }
        })

        result = create_cv_strategy(cfg)

        # アサーション
        mock_import_class.assert_called_once_with("sklearn.model_selection", "KFold")
        mock_cv_class.assert_called_once_with(n_splits=3, shuffle=False)
        assert result == mock_cv_instance

    @patch('src.mlops.utils.cv_utils.import_class')
    def test_time_series_split_creation(self, mock_import_class):
        """TimeSeriesSplit作成テスト"""
        # Mock設定
        mock_cv_class = Mock()
        mock_cv_instance = Mock()
        mock_cv_class.return_value = mock_cv_instance
        mock_import_class.return_value = mock_cv_class

        cfg = DictConfig({
            "evaluation": {
                "cv_strategy": {
                    "module": "sklearn.model_selection",
                    "class": "TimeSeriesSplit",
                    "params": {
                        "n_splits": 4,
                        "test_size": 100
                    }
                }
            }
        })

        result = create_cv_strategy(cfg)

        # アサーション
        mock_import_class.assert_called_once_with("sklearn.model_selection", "TimeSeriesSplit")
        mock_cv_class.assert_called_once_with(n_splits=4, test_size=100)
        assert result == mock_cv_instance

    @patch('src.mlops.utils.cv_utils.import_class')
    def test_shuffle_split_creation(self, mock_import_class):
        """ShuffleSplit作成テスト"""
        # Mock設定
        mock_cv_class = Mock()
        mock_cv_instance = Mock()
        mock_cv_class.return_value = mock_cv_instance
        mock_import_class.return_value = mock_cv_class

        cfg = DictConfig({
            "evaluation": {
                "cv_strategy": {
                    "module": "sklearn.model_selection",
                    "class": "ShuffleSplit",
                    "params": {
                        "n_splits": 10,
                        "test_size": 0.2,
                        "random_state": 123
                    }
                }
            }
        })

        result = create_cv_strategy(cfg)

        # アサーション
        mock_import_class.assert_called_once_with("sklearn.model_selection", "ShuffleSplit")
        mock_cv_class.assert_called_once_with(n_splits=10, test_size=0.2, random_state=123)
        assert result == mock_cv_instance

    @patch('src.mlops.utils.cv_utils.import_class')
    def test_group_kfold_creation(self, mock_import_class):
        """GroupKFold作成テスト"""
        # Mock設定
        mock_cv_class = Mock()
        mock_cv_instance = Mock()
        mock_cv_class.return_value = mock_cv_instance
        mock_import_class.return_value = mock_cv_class

        cfg = DictConfig({
            "evaluation": {
                "cv_strategy": {
                    "module": "sklearn.model_selection",
                    "class": "GroupKFold",
                    "params": {
                        "n_splits": 3
                    }
                }
            }
        })

        result = create_cv_strategy(cfg)

        # アサーション
        mock_import_class.assert_called_once_with("sklearn.model_selection", "GroupKFold")
        mock_cv_class.assert_called_once_with(n_splits=3)
        assert result == mock_cv_instance

    @patch('src.mlops.utils.cv_utils.import_class')
    def test_cv_creation_with_empty_params(self, mock_import_class):
        """空のパラメータでのCV作成テスト"""
        # Mock設定
        mock_cv_class = Mock()
        mock_cv_instance = Mock()
        mock_cv_class.return_value = mock_cv_instance
        mock_import_class.return_value = mock_cv_class

        cfg = DictConfig({
            "evaluation": {
                "cv_strategy": {
                    "module": "sklearn.model_selection",
                    "class": "KFold",
                    "params": {}
                }
            }
        })

        result = create_cv_strategy(cfg)

        # アサーション
        mock_import_class.assert_called_once_with("sklearn.model_selection", "KFold")
        mock_cv_class.assert_called_once_with()  # パラメータなしで呼び出し
        assert result == mock_cv_instance

    @patch('src.mlops.utils.cv_utils.import_class')
    def test_cv_creation_without_params_key(self, mock_import_class):
        """paramsキーがない場合のCV作成テスト"""
        # Mock設定
        mock_cv_class = Mock()
        mock_cv_instance = Mock()
        mock_cv_class.return_value = mock_cv_instance
        mock_import_class.return_value = mock_cv_class

        cfg = DictConfig({
            "evaluation": {
                "cv_strategy": {
                    "module": "sklearn.model_selection",
                    "class": "StratifiedKFold"
                    # paramsキーなし
                }
            }
        })

        result = create_cv_strategy(cfg)

        # アサーション
        mock_import_class.assert_called_once_with("sklearn.model_selection", "StratifiedKFold")
        mock_cv_class.assert_called_once_with()  # デフォルトでは空の辞書がunpack
        assert result == mock_cv_instance

    @patch('src.mlops.utils.cv_utils.import_class')
    def test_import_class_failure(self, mock_import_class):
        """import_classが失敗する場合のテスト"""
        # Mock設定（import_classがImportErrorを発生）
        mock_import_class.side_effect = ImportError("Cannot import NonexistentCV")

        cfg = DictConfig({
            "evaluation": {
                "cv_strategy": {
                    "module": "nonexistent.module",
                    "class": "NonexistentCV",
                    "params": {"n_splits": 5}
                }
            }
        })

        # ImportErrorが伝播することを確認
        with pytest.raises(ImportError) as excinfo:
            create_cv_strategy(cfg)

        assert "Cannot import NonexistentCV" in str(excinfo.value)
        mock_import_class.assert_called_once_with("nonexistent.module", "NonexistentCV")

    @patch('src.mlops.utils.cv_utils.import_class')
    def test_cv_class_instantiation_failure(self, mock_import_class):
        """CVクラスのインスタンス化が失敗する場合のテスト"""
        # Mock設定
        mock_cv_class = Mock()
        mock_cv_class.side_effect = TypeError("Invalid parameters")
        mock_import_class.return_value = mock_cv_class

        cfg = DictConfig({
            "evaluation": {
                "cv_strategy": {
                    "module": "sklearn.model_selection",
                    "class": "StratifiedKFold",
                    "params": {
                        "invalid_param": "invalid_value"
                    }
                }
            }
        })

        # TypeErrorが伝播することを確認
        with pytest.raises(TypeError) as excinfo:
            create_cv_strategy(cfg)

        assert "Invalid parameters" in str(excinfo.value)
        mock_import_class.assert_called_once_with("sklearn.model_selection", "StratifiedKFold")
        mock_cv_class.assert_called_once_with(invalid_param="invalid_value")

    def test_real_cv_strategy_creation(self):
        """実際のCV戦略作成テスト"""
        cfg = DictConfig({
            "evaluation": {
                "cv_strategy": {
                    "module": "sklearn.model_selection",
                    "class": "StratifiedKFold",
                    "params": {
                        "n_splits": 3,
                        "shuffle": True,
                        "random_state": 42
                    }
                }
            }
        })

        cv_strategy = create_cv_strategy(cfg)

        # 実際のStratifiedKFoldインスタンスが作成されることを確認
        from sklearn.model_selection import StratifiedKFold
        assert isinstance(cv_strategy, StratifiedKFold)
        assert cv_strategy.n_splits == 3
        assert cv_strategy.shuffle == True
        assert cv_strategy.random_state == 42

    def test_real_kfold_creation(self):
        """実際のKFold作成テスト"""
        cfg = DictConfig({
            "evaluation": {
                "cv_strategy": {
                    "module": "sklearn.model_selection",
                    "class": "KFold",
                    "params": {
                        "n_splits": 4,
                        "shuffle": False
                    }
                }
            }
        })

        cv_strategy = create_cv_strategy(cfg)

        # 実際のKFoldインスタンスが作成されることを確認
        from sklearn.model_selection import KFold
        assert isinstance(cv_strategy, KFold)
        assert cv_strategy.n_splits == 4
        assert cv_strategy.shuffle == False

    def test_missing_evaluation_config(self):
        """evaluation設定が存在しない場合のテスト"""
        cfg = DictConfig({
            "other_config": {"param": "value"}
            # evaluation設定なし
        })

        with pytest.raises(AttributeError):
            create_cv_strategy(cfg)

    def test_missing_cv_strategy_config(self):
        """cv_strategy設定が存在しない場合のテスト"""
        cfg = DictConfig({
            "evaluation": {
                "other_param": "value"
                # cv_strategy設定なし
            }
        })

        with pytest.raises(AttributeError):
            create_cv_strategy(cfg)

    def test_missing_module_config(self):
        """module設定が存在しない場合のテスト"""
        cfg = DictConfig({
            "evaluation": {
                "cv_strategy": {
                    "class": "StratifiedKFold",
                    "params": {"n_splits": 5}
                    # module設定なし
                }
            }
        })

        with pytest.raises(AttributeError):
            create_cv_strategy(cfg)

    def test_missing_class_config(self):
        """class設定が存在しない場合のテスト"""
        cfg = DictConfig({
            "evaluation": {
                "cv_strategy": {
                    "module": "sklearn.model_selection",
                    "params": {"n_splits": 5}
                    # class設定なし
                }
            }
        })

        with pytest.raises(KeyError):
            create_cv_strategy(cfg)