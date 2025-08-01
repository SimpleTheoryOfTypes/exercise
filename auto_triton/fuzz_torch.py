import random
from typing import Tuple, List, Dict, Callable, Optional
import time

# ───────────────────────── 1.  TensorType ──────────
class TensorType:
    def __init__(self, shape: Tuple[int, ...], dtype: str = "float32"):
        self.shape = shape
        self.dtype = dtype
    @property
    def is_float(self): return "float" in self.dtype
    @property
    def rank(self): return len(self.shape)
    def __repr__(self):
        return f"TensorType(shape={self.shape}, dtype={self.dtype})"

# ───────────────────────── 2.  Unary ops ───────────
def op_relu(x, t, env):      return f"{x} = torch.relu({x})", x, t
def op_sigmoid(x, t, env):      return f"{x} = torch.sigmoid({x})", x, t
def op_tanh(x, t, env):      return f"{x} = torch.tanh({x})", x, t
def op_add_scalar(x, t, env):return f"{x} = {x} + 1.0",        x, t
def op_mul_scalar(x, t, env):return f"{x} = {x} * 2.0",        x, t
def op_clone(x, t, env):     return f"{x} = {x}.clone()",      x, t
def op_flatten(x, t, env):   return f"{x} = {x}.view(-1)",     x, TensorType((-1,), t.dtype)
def op_softmax(x, t, env):   return f"{x} = torch.softmax({x}, dim=-1)", x, t

def op_transpose(x, t, env):
    if t.rank == 2:
        return (
            f"{x} = {x}.transpose(0, 1)",
            x,
            TensorType((t.shape[1], t.shape[0]), t.dtype),
        )
    return None

# ───────────────────────── 3.  Binary op: matmul ──
def op_matmul(x: str, tx: TensorType, env: Dict[str, TensorType]):
    for y, ty in env.items():
        if y == x or ty.dtype != tx.dtype:
            continue
        # 2-D @ 2-D
        if tx.rank == 2 and ty.rank == 2 and tx.shape[1] == ty.shape[0]:
            return (
                f"{x} = torch.matmul({x}, {y})",
                x,
                TensorType((tx.shape[0], ty.shape[1]), tx.dtype),
            )
        # 3-D @ 2-D
        if tx.rank == 3 and ty.rank == 2 and tx.shape[2] == ty.shape[0]:
            return (
                f"{x} = torch.matmul({x}, {y})",
                x,
                TensorType((tx.shape[0], tx.shape[1], ty.shape[1]), tx.dtype),
            )
    return None

def op_einsum(x: str, tx: TensorType, env: Dict[str, TensorType]):
    for y, ty in env.items():
        if y == x or ty.dtype != tx.dtype:
            continue
        # 4-D @ 2-D: torch.einsum("bijk,kl->bijl", A, B)
        if tx.rank == 4 and ty.rank == 2 and tx.shape[-1] == ty.shape[0]:
            return (
                f"{x} = torch.einsum(\"bijk,kl->bijl\", {x}, {y})",
                x,
                TensorType((tx.shape[0], tx.shape[1], ty.shape[1]), tx.dtype),
            )
    return None
         

OP_REGISTRY: List[Callable] = [
    op_relu,
    op_add_scalar,
    op_mul_scalar,
    op_clone,
    op_flatten,
    op_transpose,
    op_matmul,
    op_sigmoid,
    op_tanh,
    op_softmax,
    op_einsum
]

# ───────────────────────── 4.  Kernel generator ───
def generate_compositional_kernel(
    *,
    depth: int = 12,
    X: Optional[int] = None,
    N: Optional[int] = None,
    M: Optional[int] = None,
    K: Optional[int] = None,
    L: Optional[int] = None,
) -> str:
    """
    Builds a random kernel.  If N/M/K/L are not given, they are drawn
    uniformly from [16 … 256] so matmul is always feasible.
    """

    # Draw dimensions if not supplied
    X = X or random.randint(16, 128)
    N = N or random.randint(16, 128)
    M = M or random.randint(16, 128)
    K = K or random.randint(16, 512)
    L = L or random.randint(16, 128)
    
    env: Dict[str, TensorType] = {}
    lines: List[str] = []
    input_lines: List[str] = []

    # t0 : (X, N, M, K) <= torch.einsum
    t0 = TensorType((X, N, M, K), "float32")
    input_lines.append(f"t0 = torch.randn(({X}, {M}, {K}), dtype=torch.float32")
    env["t0"] = t0

    # t1 : (N, M, K)
    t1 = TensorType((N, M, K), "float32")
    input_lines.append(f"t1 = torch.randn(({N}, {M}, {K}), dtype=torch.float32)")
    env["t1"] = t1

    # t2 : (K, L)
    t2 = TensorType((K, L), "float32")
    input_lines.append(f"t2 = torch.randn(({K}, {L}), dtype=torch.float32)")
    env["t2"] = t2

    var_idx = 2  # we already have t1, t2

    for _ in range(depth):
        name, ttype = random.choice(list(env.items()))
        valid = [op for op in OP_REGISTRY if op(name, ttype, env) is not None]
        if not valid:
            continue
        op = random.choice(valid)
        stmt, _, new_ty = op(name, ttype, env)
        var_idx += 1
        new_name = f"t{var_idx}"
        rhs = stmt.split("=", 1)[1].strip()
        lines.append(f"{new_name} = {rhs}")
        env[new_name] = new_ty

    return "\n".join(input_lines), "\n".join(lines)

def generate_compositional_kernel2(
    *,
    depth: int = 12,
    X: Optional[int] = None,
    N: Optional[int] = None,
    M: Optional[int] = None,
    K: Optional[int] = None,
    L: Optional[int] = None,
) -> str:
    # ─── random dimensions (16–256/512) ──────────────────────
    X = X or random.randint(16, 128)
    N = N or random.randint(16, 128)
    M = M or random.randint(16, 128)
    K = K or random.randint(16, 512)
    L = L or random.randint(16, 128)

    active_env: Dict[str, TensorType] = {}
    all_seen:   Dict[str, TensorType] = {}
    lines: List[str] = []
    input_lines: List[str] = []

    # seed tensors
    input_lines.append(f"t0 = torch.randn(({X}, {N}, {M}, {K}), dtype=torch.float32)")
    active_env["t0"] = all_seen["t0"] = TensorType((X, N, M, K), "float32")

    input_lines.append(f"t1 = torch.randn(({N}, {M}, {K}), dtype=torch.float32)")
    active_env["t1"] = all_seen["t1"] = TensorType((N, M, K), "float32")

    input_lines.append(f"t2 = torch.randn(({K}, {L}), dtype=torch.float32)")
    active_env["t2"] = all_seen["t2"] = TensorType((K, L), "float32")

    var_idx = 2  # we have t0..t2

    # ─── mutation loop ───────────────────────────────────────
    for _ in range(depth):
        if not active_env:                       # nothing left to mutate
            break

        x, tx = random.choice(list(active_env.items()))
        valid = [op for op in OP_REGISTRY if op(x, tx, active_env) is not None]
        if not valid:
            # no op works for this tensor – consume it anyway
            active_env.pop(x)
            continue

        op = random.choice(valid)
        stmt, _, new_ty = op(x, tx, active_env)

        # Determine which tensors were consumed
        consumed = [x]
        if op in (op_matmul, op_einsum):         # binary ops
            # Identify y by parsing the stmt; second token after comma/space
            y = stmt.split(",", 1)[1].split(")", 1)[0].strip()
            consumed.append(y)

        # Remove consumed tensors from active_env
        for c in consumed:
            active_env.pop(c, None)

        # Produce new tensor
        var_idx += 1
        new_name = f"t{var_idx}"
        rhs = stmt.split("=", 1)[1].strip()
        lines.append(f"{new_name} = {rhs}")
        active_env[new_name] = all_seen[new_name] = new_ty

    # ─── final tensor referencing EVERY tensor seen ──────────
    t_last = new_name
    #lines.append(f"combo = torch.tensor(0.0, dtype={t_last}.dtype)")
    #for name in all_seen:
    #    lines.append(f"combo = combo + {name}.mean().to(combo.dtype)")
    #lines.append(f"t_out = {t_last} + combo * 0")
    lines.append(f"return {t_last}")

    return '\n'.join(lines), '\n'.join(input_lines)

template = """
import torch
import torch.nn as nn

class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()

    def forward(self, A, B, C):
        {your_generated_code}

def get_inputs():
    {your_generated_inputs}
    return [t0, t1, t2]

def get_init_inputs():
    return []  # No special initialization inputs needed
"""

# ───────────────────────── 5.  Demo CLI ───────────
if __name__ == "__main__":

    seed = int(time.time())
    random.seed(seed)

    import torch, random
    fuzzed_code = set()
    for i in range(9):
        print(f"\n# ── Kernel {i+1} ──")
        gen_inputs, gen_code = generate_compositional_kernel2(depth=10)
        fuzzed_code.add((gen_inputs, gen_code))
        print(gen_code)
    print(f"unique kernels: {len(fuzzed_code)}")
    print(template)
