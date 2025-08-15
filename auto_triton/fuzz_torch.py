"""
compositional_fuzzer.py  –  emits full PyTorch files

$ python compositional_fuzzer.py > fuzz_model.py
"""

import random, time
from typing import Tuple, List, Dict, Callable, Optional

# ───────────────────────── 1.  TensorType ──────────
class TensorType:
    def __init__(self, shape: Tuple[int, ...], dtype: str = "float32"):
        self.shape = shape
        self.dtype = dtype
    @property
    def is_float(self): return "float" in self.dtype
    @property
    def rank(self): return len(self.shape)
    def __repr__(self): return f"TensorType(shape={self.shape}, dtype={self.dtype})"

# ───────────────────────── 2.  Op definitions ──────
def op_relu(x, t, env):      return f"{x} = torch.relu({x})", x, t
def op_sigmoid(x, t, env):   return f"{x} = torch.sigmoid({x})", x, t
def op_tanh(x, t, env):      return f"{x} = torch.tanh({x})", x, t
def op_add_scalar(x, t, env):return f"{x} = {x} + 1.0",        x, t
def op_mul_scalar(x, t, env):return f"{x} = {x} * 2.0",        x, t
def op_clone(x, t, env):     return f"{x} = {x}.clone()",      x, t
def op_flatten(x, t, env):   return f"{x} = {x}.view(-1)",     x, TensorType((-1,), t.dtype)
def op_softmax(x, t, env):   return f"{x} = torch.softmax({x}, dim=-1)", x, t

def op_transpose(x, t, env):
    if t.rank == 2:
        return f"{x} = {x}.transpose(0, 1)", x, TensorType((t.shape[1], t.shape[0]), t.dtype)
    return None

def op_matmul(x: str, tx: TensorType, env: Dict[str, TensorType]):
    for y, ty in env.items():
        if y == x or ty.dtype != tx.dtype:
            continue
        if tx.rank == 2 and ty.rank == 2 and tx.shape[1] == ty.shape[0]:
            out = TensorType((tx.shape[0], ty.shape[1]), tx.dtype)
            return f"{x} = torch.matmul({x}, {y})", x, out
        if tx.rank == 3 and ty.rank == 2 and tx.shape[2] == ty.shape[0]:
            out = TensorType((tx.shape[0], tx.shape[1], ty.shape[1]), tx.dtype)
            return f"{x} = torch.matmul({x}, {y})", x, out
    return None

def op_einsum(x: str, tx: TensorType, env: Dict[str, TensorType]):
    for y, ty in env.items():
        if y == x or ty.dtype != tx.dtype:
            continue
        if tx.rank == 4 and ty.rank == 2 and tx.shape[-1] == ty.shape[0]:
            out = TensorType((tx.shape[0], tx.shape[1], ty.shape[1]), tx.dtype)
            code = f'{x} = torch.einsum("bijk,kl->bijl", {x}, {y})'
            return code, x, out
    return None

OP_REGISTRY: List[Callable] = [
    op_relu, op_add_scalar, op_mul_scalar, op_clone, op_flatten,
    op_transpose, op_matmul, op_sigmoid, op_tanh, op_softmax, op_einsum
]

# ───────────────────────── 3.  Kernel generator ───
def generate_kernel(depth: int = 12) -> Tuple[str, str]:
    # random dimensions
    X,N,M,K,L = (random.randint(16,128) for _ in range(5))

    active: Dict[str, TensorType] = {}
    all_seen: Dict[str, TensorType] = {}
    inputs, body = [], []

    # seed tensors (ensure matmul / einsum partners)
    inputs.append(f"t0 = torch.randn(({X}, {N}, {M}, {K}), dtype=torch.float32)")
    active["t0"] = all_seen["t0"] = TensorType((X,N,M,K),"float32")

    inputs.append(f"t1 = torch.randn(({N}, {M}, {K}), dtype=torch.float32)")
    active["t1"] = all_seen["t1"] = TensorType((N,M,K),"float32")

    inputs.append(f"t2 = torch.randn(({K}, {L}), dtype=torch.float32)")
    active["t2"] = all_seen["t2"] = TensorType((K,L),"float32")

    idx = 2
    for _ in range(depth):
        if not active: break
        x, tx = random.choice(list(active.items()))
        valid = [op for op in OP_REGISTRY if op(x, tx, active) is not None]
        if not valid:
            active.pop(x); continue
        op = random.choice(valid)
        stmt, _, new_ty = op(x, tx, active)

        # figure out consumed tensors
        consumed = [x]
        if op in (op_matmul, op_einsum):
            y = stmt.split(",",1)[1].split(")",1)[0].strip()
            consumed.append(y)
        for c in consumed:
            active.pop(c, None)

        idx += 1
        new_name = f"t{idx}"
        rhs = stmt.split("=",1)[1].strip()
        body.append(f"{new_name} = {rhs}")
        active[new_name] = all_seen[new_name] = new_ty

    t_last = new_name
    body.append(f"combo = torch.tensor(0.0, dtype={t_last}.dtype)")
    for name in all_seen:
        body.append(f"combo = combo + {name}.mean().to(combo.dtype)")
    body.append(f"t_out = {t_last} + combo * 0")
    body.append("return t_out")

    return "\n".join(inputs), "\n".join(body)

# ───────────────────────── 4.  Template ───────────
TEMPLATE = """
import torch
import torch.nn as nn

class Model(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, t0, t1, t2):
{body_indent}

def get_inputs():
{inputs_indent}
    return [t0, t1, t2]

def get_init_inputs():
    return []
"""

def make_module_source(inputs: str, body: str) -> str:
    indent = lambda s, n: "\n".join(" " * n + ln for ln in s.splitlines())
    return TEMPLATE.format(
        body_indent   = indent(body, 8),
        inputs_indent = indent(inputs, 4)
    )

# ───────────────────────── 5.  CLI demo ───────────
if __name__ == "__main__":
    random.seed(int(time.time()))
    for i in range(3):
        ins, bod = generate_kernel(depth=10)
        print(f"# ── Generated module {i+1} ──")
        print(make_module_source(ins, bod))

