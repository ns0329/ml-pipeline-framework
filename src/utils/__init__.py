"""汎用ユーティリティパッケージ"""

from .core_utils import (
    import_class,
    resolve_function_references,
    get_dataset_name,
    detect_task_type,
)

__all__ = [
    'import_class',
    'resolve_function_references',
    'get_dataset_name',
    'detect_task_type',
]