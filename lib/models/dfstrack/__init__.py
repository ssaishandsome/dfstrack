try:
    from .dfstrack import build_dfstrack
except ModuleNotFoundError as exc:
    if exc.name != "transformers":
        raise
    _IMPORT_ERROR = exc

    def build_dfstrack(*args, **kwargs):
        raise ModuleNotFoundError(
            "build_dfstrack requires the optional dependency 'transformers' to be installed"
        ) from _IMPORT_ERROR
