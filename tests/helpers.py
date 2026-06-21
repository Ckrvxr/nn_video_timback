import sys
from types import ModuleType
from pathlib import Path
from unittest.mock import MagicMock


_project_root = str(Path(__file__).resolve().parent.parent)


def setup_sys_path():
    if _project_root not in sys.path:
        sys.path.insert(0, _project_root)


class MockTriton(ModuleType):
    def __init__(self):
        super().__init__('triton')
        import importlib.util
        self.__spec__ = importlib.util.spec_from_loader('triton', loader=None)
        self.language = MagicMock()

    @classmethod
    def jit(cls, *args, **kwargs):
        if len(args) == 1 and callable(args[0]):
            return args[0]
        def decorator(f):
            return f
        return decorator


def mock_triton():
    sys.modules['triton'] = MockTriton()


def mock_mamba_ssm():
    import torch

    class _IdentityMambaSSM:
        def __init__(self, d_model=16, d_state=8, *a, **kw):
            self.d_model = d_model
            self.d_state = d_state
        def __call__(self, x, h_state=None):
            return x

    _sm = MagicMock()
    _sm.Mamba = _IdentityMambaSSM
    _sm.selective_scan_fn = lambda *a, **kw: (a[0].clone() if a and isinstance(a[0], torch.Tensor) else torch.zeros(1, 1, 1), torch.zeros(1))
    _sm.mamba_inner_fn = lambda x, *a, **kw: x.clone() if isinstance(x, torch.Tensor) else torch.zeros(1, 1, 1)
    sys.modules['mamba_ssm'] = _sm
    sys.modules['mamba_ssm.ops'] = MagicMock()
    sys.modules['mamba_ssm.ops.selective_scan_interface'] = _sm
    sys.modules['mamba_ssm.ops.triton'] = MagicMock()
    sys.modules['mamba_ssm.ops.triton.layer_norm'] = MagicMock()
