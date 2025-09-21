"""動的import・クラス解決ユーティリティ"""
import importlib


def import_class(module_name: str, class_name: str):
    """動的にクラスをインポート"""
    try:
        module = importlib.import_module(module_name)
        return getattr(module, class_name)
    except (ImportError, AttributeError) as e:
        raise ImportError(f"Failed to import {class_name} from {module_name}: {e}")


def resolve_function_references(params):
    """任意の関数参照を動的解決（プロジェクト横断で再利用可能）"""
    for key, value in params.items():
        if isinstance(value, str) and '.' in value and not value.startswith(('http', 'file', '/', './')):
            try:
                # "sklearn.feature_selection.f_classif" → 実関数に変換
                module_path, func_name = value.rsplit('.', 1)
                module = importlib.import_module(module_path)
                params[key] = getattr(module, func_name)
            except (ImportError, AttributeError):
                # インポートに失敗した場合は通常の文字列のまま
                pass
    return params