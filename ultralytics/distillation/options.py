"""Public CLQ-KD options; explicit train arguments override YAML defaults."""
import math
from numbers import Real
from pathlib import Path
import yaml

DEFAULTS = dict(clqkd_alpha=5.0, clqkd_beta=0.0, clqkd_teacher_temperature=1.5,
                clqkd_student_temperature=1.0, clqkd_prob_threshold=.001, clqkd_iou_threshold=.3)


def resolve_clqkd(method, values):
    enabled = values.get('clqkd', False)
    if type(enabled) is not bool:
        raise ValueError('clqkd must be a boolean')
    if enabled and method in ('baseline', 'clqkd'):
        raise ValueError('clqkd=True is only for combination methods')
    active = enabled or method == 'clqkd'
    if not active and any(values.get(k) is not None for k in DEFAULTS):
        raise ValueError('CLQ-KD parameters require an enabled CLQ-KD objective')
    config = yaml.safe_load((Path(__file__).resolve().parents[2] / 'configs/distillation.yaml').read_text(encoding='utf8'))
    defaults = {**DEFAULTS, **{k: v for k, v in config['clqkd'].items() if k in DEFAULTS}}
    if enabled:
        defaults['clqkd_beta'] = config.get('clqkd_combination', {}).get('clqkd_beta', 1.0)
    out = {}
    if active:
        for k, default in defaults.items():
            v = values.get(k)
            v = default if v is None else v
            if isinstance(v, bool) or not isinstance(v, Real) or not math.isfinite(v):
                raise ValueError(f'{k} must be a finite number')
            if ('temperature' in k and v <= 0) or v < 0 or ('threshold' in k and v > 1):
                raise ValueError(f'Invalid {k}: {v}')
            out[k] = float(v)
    return enabled, out
