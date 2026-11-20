# Ultralytics AGPL-3.0 License - https://ultralytics.com/license
import hashlib
import json
import math
import weakref
import torch
from torch import nn
from .loss import CSAKDModules
CSAKD_ALPHA = 300.0
CSAKD_BETA = 10.0
KD_BATCH_MODE = 'full'
YOLO11_FROM = [-1] * 23
for (_index, _skip) in ((12, 6), (15, 4), (18, 13), (21, 10)):
    YOLO11_FROM[_index] = [-1, _skip]

def validate_csakd_weight(value, name='CSAKD weight'):
    value = float(value)
    if not math.isfinite(value) or value < 0:
        raise ValueError(f'{name} must be finite and nonnegative')
    return value

def validate_batch_mode(value):
    if value != KD_BATCH_MODE:
        raise ValueError('Peer CSAKD requires full actual-batch computation, without checkpointing or sample chunks')
    return KD_BATCH_MODE

def model_architecture_signature(model):
    """Compare every nested type/repeat, state shape and operator setting.

    Weight values, runtime caches, training mode and requires_grad are excluded:
    the pretrained frozen teacher and random trainable student differ in those.
    Root nc is optional load-time metadata, validated against Detect/YAML below
    and omitted only at the root; nested operator nc remains strictly compared.
    Named nested modules reveal internal differences even when outer-node
    output channels happen to agree.
    """
    operator_fields = ('kernel_size', 'stride', 'padding', 'dilation', 'groups', 'in_channels', 'out_channels', 'num_features', 'eps', 'momentum', 'affine', 'track_running_stats', 'size', 'scale_factor', 'mode', 'align_corners', 'dim', 'd', 'nc', 'nl', 'reg_max', 'legacy', 'end2end', 'num_heads', 'head_dim', 'key_dim', 'scale', 'add', 'c', 'attn_ratio')

    def value_signature(value):
        if isinstance(value, torch.Tensor):
            return {'shape': list(value.shape), 'values': value.detach().cpu().tolist()}
        if isinstance(value, (list, tuple)):
            return [value_signature(item) for item in value]
        if isinstance(value, (str, int, float, bool)) or value is None:
            return value
        raise TypeError(f'Unsupported architecture attribute type: {type(value).__name__}')
    modules = []
    for (name, module) in model.named_modules():
        # Older public checkpoints omit this root flag; its runtime default is False.
        modules.append(dict(name=name, type=f'{type(module).__module__}.{type(module).__qualname__}', children=list(module._modules), parameters={key: list(value.shape) for (key, value) in module.named_parameters(recurse=False)}, buffers={key: list(value.shape) for (key, value) in module.named_buffers(recurse=False)}, settings={key: value_signature(getattr(module, key)) for key in operator_fields if hasattr(module, key) and (not (name == '' and key == 'nc'))}, graph_from=value_signature(getattr(module, 'f', None))))
        if name == '':
            modules[-1]['settings']['end2end'] = getattr(module, 'end2end', False)
    return dict(modules=modules, state_shapes={key: list(value.shape) for (key, value) in model.state_dict().items()})

def validate_peer_architecture(student, teacher):
    for (role, model) in (('student', student), ('teacher', teacher)):
        head_nc = getattr(model.model[-1], 'nc', None)
        yaml_config = getattr(model, 'yaml', None)
        yaml_nc = yaml_config.get('nc') if isinstance(yaml_config, dict) else None
        if type(head_nc) is not int or type(yaml_nc) is not int or head_nc <= 0 or (yaml_nc != head_nc):
            raise ValueError(f'CSAKD {role} Detect head and YAML nc must agree and be positive')
        if hasattr(model, 'nc') and (type(model.nc) is not int or model.nc != head_nc):
            raise ValueError(f'CSAKD {role} root nc metadata must agree with Detect head and YAML nc')
        if any((value.is_floating_point() and value.dtype != torch.float32 for value in model.state_dict().values())):
            raise ValueError(f'CSAKD {role} must use FP32 model parameters and buffers')
    student_signature = model_architecture_signature(student)
    teacher_signature = model_architecture_signature(teacher)
    if student_signature != teacher_signature:
        raise ValueError('Peer CSAKD requires identical full internal types, repeat depth, operator settings and state shapes')
    encoded = json.dumps(student_signature, sort_keys=True, separators=(',', ':')).encode()
    return dict(full_internal_architecture_sha256=hashlib.sha256(encoded).hexdigest(), nested_module_count=len(student_signature['modules']), state_tensor_count=len(student_signature['state_shapes']))

def _normal_features(model, images):
    """Run only nonDetect graph, retaining every outer-node result."""
    (saved, value) = ([], images)
    for module in model.model[:-1]:
        if module.f != -1:
            value = saved[module.f] if isinstance(module.f, int) else [value if index == -1 else saved[index] for index in module.f]
        value = module(value)
        if not isinstance(value, torch.Tensor) or value.ndim != 4:
            raise ValueError('CSAKD nonDetect nodes must all return BCHW features')
        saved.append(value)
    return saved

class CSAKDCapture(CSAKDModules):
    """Parameter-free peer alignment; detector ownership uses weak references.

    Every node simultaneously processes the full actual batch. Ordinary teacher
    targets are no_grad, whereas fixed teacher cross operators retain autograd
    to student features. No sample splitting or activation checkpoint is used.
    """
    indices = list(range(23))

    def __init__(self, student, teacher, csakd_alpha=CSAKD_ALPHA, csakd_beta=CSAKD_BETA, kd_batch_mode=KD_BATCH_MODE):
        for model in (student, teacher):
            head = model.model[-1]
            if len(model.model) != 24 or list(head.f) != [16, 19, 22] or head.nc != student.model[-1].nc or getattr(head, 'end2end', False):
                raise ValueError('Expected native 24-module YOLO11 model')
            if [module.i for module in model.model[:-1]] != self.indices:
                raise ValueError('CSAKD outer node indices must be contiguous 0..22')
            if [module.f for module in model.model[:-1]] != YOLO11_FROM:
                raise ValueError('CSAKD requires the native YOLO11 outer graph')
        if any((module.training for module in teacher.modules())) or any((p.requires_grad for p in teacher.parameters())):
            raise ValueError('CSAKD teacher must be entirely frozen and eval')
        architecture = validate_peer_architecture(student, teacher)
        modes = {module: module.training for model in (student, teacher) for module in model.modules()}
        try:
            student.eval()
            teacher.eval()
            dummy = torch.zeros(1, 3, 64, 64, device=next(student.parameters()).device)
            with torch.no_grad():
                shapes_s = [tuple(x.shape) for x in _normal_features(student, dummy)]
                shapes_t = [tuple(x.shape) for x in _normal_features(teacher, dummy)]
        finally:
            for (module, training) in modes.items():
                module.training = training
        if shapes_s != shapes_t:
            raise ValueError('CSAKD peer feature shapes differ; projection and interpolation are forbidden')
        super().__init__([shape[1] for shape in shapes_s], [shape[1] for shape in shapes_t])
        object.__setattr__(self, '_student_ref', weakref.ref(student))
        object.__setattr__(self, '_teacher_ref', weakref.ref(teacher))
        self.csakd_alpha = validate_csakd_weight(csakd_alpha, 'csakd_alpha')
        self.csakd_beta = validate_csakd_weight(csakd_beta, 'csakd_beta')
        self.kd_batch_mode = validate_batch_mode(kd_batch_mode)
        self.architecture = architecture
        (self._hook, self.enabled, self.features) = (None, False, None)
        self.last_shapes = {}

    def install_hooks(self):
        if self._hook is not None:
            return
        owner_ref = weakref.ref(self)
        hooks = []
        for index in self.indices:

            def capture(_module, _inputs, output, node_index=index):
                owner = owner_ref()
                if owner is None or not owner.enabled:
                    return
                if owner.features[node_index] is not None:
                    raise RuntimeError(f'Duplicate CSAKD capture at node {node_index}')
                if not isinstance(output, torch.Tensor) or output.ndim != 4:
                    raise ValueError('CSAKD requires every nonDetect BCHW output')
                owner.features[node_index] = output
            hooks.append(self._student_ref().model[index].register_forward_hook(capture))
        self._hook = hooks

    def remove_hooks(self):
        self.disable_capture()
        if self._hook is not None:
            for hook in self._hook:
                hook.remove()
            self._hook = None

    def enable_capture(self):
        if self._hook is None:
            raise RuntimeError('Capture hooks must be installed after EMA construction')
        (self.features, self.last_shapes, self.enabled) = ([None] * len(self.indices), {}, True)

    def disable_capture(self):
        (self.features, self.enabled) = (None, False)

    def _assert_teacher(self):
        teacher = self._teacher_ref()
        if teacher is None or any((module.training for module in teacher.modules())) or any((p.requires_grad for p in teacher.parameters())):
            raise RuntimeError('Teacher must remain completely frozen and eval')
        return teacher

    def normal_teacher_features(self, images):
        teacher = self._assert_teacher()
        with torch.no_grad():
            return _normal_features(teacher, images)

    def cross_teacher_features(self, aligned):
        teacher = self._assert_teacher()
        if len(aligned) != len(self.indices):
            raise ValueError('CSAKD cross graph requires all 23 identity-aligned student features')
        cross = [aligned[0]]
        for (index, module) in enumerate(teacher.model[1:-1], start=1):
            value = aligned[index - 1]
            if module.f != -1:
                value = cross[module.f] if isinstance(module.f, int) else [value if previous == -1 else cross[previous] for previous in module.f]
            value = module(value)
            if value.shape != aligned[index].shape:
                raise ValueError(f'CSAKD cross geometry mismatch at outer node {index}')
            cross.append(value)
        return cross

    def _full_batch_parts(self, images, *student_features):
        target = self.normal_teacher_features(images)
        callback = getattr(self, 'normal_target_callback', None)
        if callback is not None:
            callback([target[i] for i in self._teacher_ref().model[-1].f])
        aligned = self.align(student_features)
        cross = self.cross_teacher_features(aligned)
        return self.per_sample_parts(aligned, target, cross)

    def forward(self, images, student_features, use_checkpoint=False, return_parts=False):
        if use_checkpoint is not False:
            raise ValueError('Activation checkpointing is forbidden in the approved full-batch peer CSAKD protocol')
        self._assert_teacher()
        if images.ndim != 4 or images.shape[0] <= 0 or len(student_features) != len(self.indices):
            raise ValueError('CSAKD needs actual images and all student feature levels')
        if any((not isinstance(x, torch.Tensor) or x.ndim != 4 or x.shape[0] != images.shape[0] for x in student_features)):
            raise ValueError('CSAKD captured student batch geometry mismatch')
        self.last_shapes = dict(student=[list(x.shape) for x in student_features], teacher=[[x.shape[0], channels, *x.shape[2:]] for (x, channels) in zip(student_features, self.teacher_channels)])
        stats = self._full_batch_parts(images, *student_features)
        (agka, cross, parts) = self.aggregate_parts(*stats)
        return (agka, cross, parts) if return_parts else (agka, cross)
