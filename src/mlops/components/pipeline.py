"""パイプライン構築コンポーネント"""
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from omegaconf import DictConfig
import inspect
from src.utils.import_utils import import_class, resolve_function_references

# サンプリング対応のためImblearn Pipelineもインポート
try:
    from imblearn.pipeline import Pipeline as ImbPipeline
    IMBLEARN_AVAILABLE = True
except ImportError:
    ImbPipeline = Pipeline
    IMBLEARN_AVAILABLE = False

# サンプリング関連の定義
SAMPLING_MODULES = ['imblearn.over_sampling', 'imblearn.under_sampling', 'imblearn.combine']
SAMPLING_CLASSES = ['SMOTE', 'RandomOverSampler', 'ADASYN', 'RandomUnderSampler', 'EditedNearestNeighbours', 'BorderlineSMOTE']

def _is_sampler_step(step_config):
    """ステップがサンプラーかどうかを判定"""
    return (step_config.module in SAMPLING_MODULES or
            step_config.get('class', '') in SAMPLING_CLASSES)


def generate_optuna_params(trial, optuna_space):
    """Optuna用パラメータを動的生成"""
    params = {}
    for param_name, space_config in optuna_space.items():
        param_type, *args = space_config
        if param_type == "int":
            params[param_name] = trial.suggest_int(param_name, *args)
        elif param_type == "float":
            params[param_name] = trial.suggest_float(param_name, *args)
        elif param_type == "categorical":
            params[param_name] = trial.suggest_categorical(param_name, *args)
    return params


def create_pipeline_step(step_config):
    """Pipeline ステップを動的作成（動的インポート方式）"""
    module_name = step_config.module
    class_name = step_config["class"]
    params = dict(step_config.get('params', {}))

    # 動的インポートでクラス取得
    step_class = import_class(module_name, class_name)

    # 汎用的な関数参照解決（任意のパラメータ・関数で利用可能）
    params = resolve_function_references(params)

    transformer = step_class(**params)

    # カラム指定がある場合はColumnTransformerでラップ
    if hasattr(step_config, 'columns') and step_config.columns != 'all':
        # OmegaConfのリストを通常のPythonリストに変換
        columns = list(step_config.columns)

        return ColumnTransformer([
            (step_config.name, transformer, columns)
        ], remainder='passthrough', sparse_threshold=0)

    return transformer


def create_pipeline(cfg: DictConfig, trial=None, best_params=None):
    """設定に基づいてPipelineを構築（Pipeline Config駆動）"""
    steps = []
    has_sampler = False

    # Pipeline Config方式のみサポート
    for step_config in cfg.pipeline.steps:
        step_name = step_config.name

        # サンプラー検出（enabled判定前に実行）
        if _is_sampler_step(step_config) and getattr(step_config, 'enabled', True):
            has_sampler = True

        # enabled フラグがFalseの場合はスキップ
        if hasattr(step_config, 'enabled') and not step_config.enabled:
            continue

        pipeline_step = create_pipeline_step(step_config)
        steps.append((step_name, pipeline_step))

    # モデル設定を直接取得
    model_class = import_class(cfg.model.module, cfg.model["class"])
    optuna_space = {k: tuple(v) for k, v in cfg.model.optuna_space.items()}

    if trial:  # Optuna最適化時
        params = generate_optuna_params(trial, optuna_space)
        print(f"    [pipeline] trial mode: params={params}")
    elif best_params:  # Optuna最適化結果適用時
        params = {**best_params}
        print(f"    [pipeline] best_params mode: params={params}")
    else:  # 通常実行時
        params = {**cfg.model.default_params}
        print(f"    [pipeline] default mode: params={params}")

    # random_stateパラメータが対応している場合のみ追加
    if "random_state" in inspect.signature(model_class.__init__).parameters:
        params["random_state"] = cfg.data.random_state

    steps.append(("classifier", model_class(**params)))

    # サンプラーがある場合はImbPipelineを使用
    if has_sampler and IMBLEARN_AVAILABLE:
        return ImbPipeline(steps)
    else:
        return Pipeline(steps)