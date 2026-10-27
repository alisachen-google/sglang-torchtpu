"""Per-rank trace of torch.distributed calls, for finding where TP ranks diverge.

SGLANG_TPU_COLLECTIVE_TRACE=<dir> makes every rank append one line per
collective to <dir>/ctrace_r<rank>_<pid>.log: a sequence number, the caller
chain outside torch, the group's ranks, and the input tensor's shape plus its
values when it is a small integer CPU tensor. Diffing the files rank against
rank (pd_disagg/spill/collective_diff.py) shows the first call where the
ranks' call sites, shapes or inputs differ; a hang follows such a call.
"""

import os
import sys
import threading
import time

import torch
import torch.distributed as dist

_OPS = (
    "all_reduce",
    "all_gather",
    "all_gather_object",
    "all_gather_into_tensor",
    "broadcast",
    "broadcast_object_list",
    "barrier",
    "gather",
    "scatter",
    "all_to_all_single",
    "reduce_scatter_tensor",
)
_SKIP = ("torch/distributed", "c10d_logger", "collective_trace", "contextlib")

_state = {"seq": 0, "fh": None, "dir": None, "groups": {}}
_lock = threading.Lock()


def _caller() -> str:
    frame = sys._getframe(3)
    parts = []
    while frame is not None and len(parts) < 5:
        filename = frame.f_code.co_filename
        if not any(s in filename for s in _SKIP):
            parts.append(
                f"{os.path.basename(filename)}:{frame.f_lineno}:{frame.f_code.co_name}"
            )
        frame = frame.f_back
    return "<".join(parts)


def _desc(value) -> str:
    if isinstance(value, torch.Tensor):
        shape = "x".join(map(str, value.shape)) or "scalar"
        text = f"{shape}:{str(value.dtype).replace('torch.', '')}@{value.device.type}"
        if (
            value.device.type == "cpu"
            and not value.is_floating_point()
            and value.numel() <= 64
        ):
            text += "=" + ",".join(map(str, value.flatten().tolist()))
        return text
    if isinstance(value, (list, tuple)):
        head = ",".join(_desc(v) for v in value[:4])
        return f"[{head}{'...' if len(value) > 4 else ''}]#{len(value)}"
    if isinstance(value, (int, str, bytes, type(None))):
        return repr(value)[:80]
    return type(value).__name__


def _group_desc(group) -> str:
    key = id(group)
    cached = _state["groups"].get(key)
    if cached is None:
        try:
            cached = ",".join(map(str, dist.get_process_group_ranks(group)))
        except Exception:  # noqa: BLE001 - a group torch cannot describe
            cached = "?"
        _state["groups"][key] = cached
    return cached


def _file():
    fh = _state["fh"]
    if fh is None:
        rank = dist.get_rank() if dist.is_initialized() else -1
        path = os.path.join(_state["dir"], f"ctrace_r{rank}_{os.getpid()}.log")
        fh = _state["fh"] = open(path, "a", buffering=1 << 16)  # noqa: SIM115
        threading.Thread(target=_flusher, args=(fh,), daemon=True).start()
    return fh


def _flusher(fh) -> None:
    while True:
        time.sleep(1.0)
        with _lock:
            fh.flush()


def _record(name: str, args, kwargs) -> None:
    group = kwargs.get("group")
    if group is None and name in ("all_reduce", "broadcast") and len(args) > 2:
        group = args[2]
    op = kwargs.get("op")
    if op is None and name == "all_reduce" and len(args) > 1:
        op = args[1]
    payload = args[0] if args else kwargs.get("tensor", kwargs.get("object_list"))
    if name in ("all_gather", "all_gather_object", "all_gather_into_tensor", "gather"):
        payload = args[1] if len(args) > 1 else kwargs.get("tensor", kwargs.get("obj"))
    with _lock:
        _state["seq"] += 1
        line = (
            f"{_state['seq']} {time.monotonic():.3f} {threading.current_thread().name} "
            f"{name}{'' if op is None else '/' + str(op).split('.')[-1]} g=[{_group_desc(group)}] "
            f"{_desc(payload)} @ {_caller()}\n"
        )
        _file().write(line)


def note(text: str) -> None:
    """Append a rank-local state line to the trace, in sequence with the calls."""
    if _state["dir"] is None:
        return
    with _lock:
        _state["seq"] += 1
        _file().write(
            f"{_state['seq']} {time.monotonic():.3f} {threading.current_thread().name} note {text}\n"
        )


def _wrap(name: str):
    original = getattr(dist, name)

    def traced(*args, **kwargs):
        _record(name, args, kwargs)
        return original(*args, **kwargs)

    traced.__name__ = name
    traced.__wrapped__ = original
    return traced


def install_from_env() -> bool:
    directory = os.environ.get("SGLANG_TPU_COLLECTIVE_TRACE", "")
    if not directory or _state["dir"] is not None:
        return False
    os.makedirs(directory, exist_ok=True)
    _state["dir"] = directory
    for name in _OPS:
        setattr(dist, name, _wrap(name))
    return True
