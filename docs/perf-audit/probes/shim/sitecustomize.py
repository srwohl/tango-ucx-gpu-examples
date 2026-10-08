"""Profiling-only: label Processor entry points with NVTX ranges, leaving the repo untouched."""
import importlib.abc
import sys


class _Label(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path, target=None):
        if fullname != "processors":
            return None
        for finder in sys.meta_path:
            if finder is self or not hasattr(finder, "find_spec"):
                continue
            spec = finder.find_spec(fullname, path, target)
            if spec is not None and spec.loader is not None:
                execute = spec.loader.exec_module

                def exec_module(module, execute=execute):
                    execute(module)
                    _wrap(module)
                spec.loader.exec_module = exec_module
                return spec
        return None


def _wrap(module):
    from cupy.cuda import nvtx

    def labelled(method):
        function = getattr(module.Processor, method)

        def wrapper(self, *args, **kwargs):
            nvtx.RangePush(f"{self.role}.{method}")
            try:
                return function(self, *args, **kwargs)
            finally:
                nvtx.RangePop()
        setattr(module.Processor, method, wrapper)
    for method in ("consume", "consume_many", "reconstruct", "reconstruct_block"):
        labelled(method)


sys.meta_path.insert(0, _Label())
