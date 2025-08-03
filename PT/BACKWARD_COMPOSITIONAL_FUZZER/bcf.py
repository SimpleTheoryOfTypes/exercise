# backward_fuzzer.py
import random
from typing import Tuple, List, Dict, NamedTuple

# ───────────────────────── 1.  Tiny tensor "type" ─────────────────────────
class TType(NamedTuple):
    shape: Tuple[int, ...]
    dtype: str = "float32"
    @property
    def rank(self): return len(self.shape)

# ───────────────────────── 2.  Op registry (shape-preserving unary) ───────
UNARY_FMTS = [
    ("relu",    lambda x: f"torch.relu({x})"),
    ("sigmoid", lambda x: f"torch.sigmoid({x})"),
    ("tanh",    lambda x: f"torch.tanh({x})"),
    # softmax valid only for rank>=1; we guard before using it
    ("softmax", lambda x: f"torch.softmax({x}, dim=-1)"),
]

def can_use_softmax(tt: TType) -> bool:
    return tt.rank >= 1

# ───────────────────────── 3.  Fresh names ────────────────────────────────
def fresh(counter: List[int]) -> str:
    counter[0] += 1
    return f"t{counter[0]}"

# ───────────────────────── 4.  Backward builder (goal → operands) ─────────
def backward_build(
    *,
    out_shape: Tuple[int, ...],
    depth: int = 8,
    min_chain: int = 3,
    seed: int | None = None,
) -> Dict[str, tuple[str, TType, List[str]]]:
    """
    Returns a graph:
        name -> (rhs_string, TType, deps_list)
    starting from 't_out' of shape `out_shape`, expanding backwards for up to `depth`
    steps; each predecessor becomes a subgoal. Branches stop and become leaves
    (torch.randn(...)) when no budget remains or by random choice.
    """
    if seed is not None:
        random.seed(seed)

    produced: Dict[str, tuple[str, TType, List[str]]] = {}  # node -> (rhs, type, deps)
    all_nodes: set[str] = set()                             # track names we referenced
    counter = [-1]                                          # will start at t0
    # Stack of subgoals to satisfy
    unresolved: List[tuple[str, TType, int]] = [("t_out", TType(out_shape), min_chain)]
    steps_left = depth

    def add_leaf_if_missing(name: str, ty: TType):
        """If a node has no definition yet, make it a torch.randn(...) leaf."""
        if name not in produced:
            produced[name] = (f"torch.randn({ty.shape}, dtype=torch.float32)", ty, [])
            all_nodes.add(name)

    while unresolved and steps_left > 0:
        name, ty, budget = unresolved.pop()
        all_nodes.add(name)

        # If the node is already defined (e.g., reached again), skip.
        if name in produced:
            steps_left -= 1
            continue

        # Choose an op family compatible with the target rank.
        # Bias towards unary to create chains; use matmul/einsum for rank-3 targets.
        if ty.rank >= 3:
            choice = random.choices(
                population=["unary", "matmul", "einsum"],
                weights=[0.5, 0.3, 0.2],
                k=1
            )[0]
        elif ty.rank == 2:
            # 2-D target: build a unary chain (or you could add 2D matmul variants)
            choice = "unary"
        elif ty.rank == 1:
            choice = "unary"
        else:
            # scalar target → unary chain on scalars doesn't change shape; just leaf it
            add_leaf_if_missing(name, ty)
            steps_left -= 1
            continue

        # Stopping rule: if we exhausted the budget for this path, materialize a leaf.
        if budget <= 0:
            add_leaf_if_missing(name, ty)
            steps_left -= 1
            continue

        if choice == "unary":
            # pick a unary op that's valid for the rank
            pool = UNARY_FMTS if can_use_softmax(ty) else UNARY_FMTS[:-1]
            op_name, fmt = random.choice(pool)

            src = fresh(counter)
            # define current node as unary(src), and push src as a subgoal
            produced[name] = (fmt(src), ty, [src])
            all_nodes.add(src)
            unresolved.append((src, ty, budget - 1))

        elif choice == "matmul" and ty.rank == 3:
            # target (b,i,j) = matmul(A:(b,i,k), B:(k,j))
            b, i, j = ty.shape
            k = random.randint(8, 128)
            A_ty = TType((b, i, k))
            B_ty = TType((k, j))
            A = fresh(counter)
            B = fresh(counter)
            produced[name] = (f"torch.matmul({A}, {B})", ty, [A, B])
            all_nodes.update([A, B])
            # push subgoals; give A some budget, B smaller (heuristic)
            unresolved.append((A, A_ty, max(0, budget - 1)))
            unresolved.append((B, B_ty, max(0, budget - 2)))

        elif choice == "einsum" and ty.rank == 3:
            # Use a 3-D einsum equivalent to matmul for clean shapes:
            #   "bik,kl->bil": A:(b,i,k), B:(k,l=j) => out:(b,i,l=j) == (b,i,j)
            b, i, j = ty.shape
            l = j
            k = random.randint(8, 128)
            A_ty = TType((b, i, k))
            B_ty = TType((k, l))
            A = fresh(counter)
            B = fresh(counter)
            produced[name] = (f'torch.einsum("bik,kl->bil", {A}, {B})', ty, [A, B])
            all_nodes.update([A, B])
            unresolved.append((A, A_ty, max(0, budget - 1)))
            unresolved.append((B, B_ty, max(0, budget - 2)))
        else:
            # Fallback: leaf
            add_leaf_if_missing(name, ty)

        steps_left -= 1

    # Materialize any remaining subgoals as leaves
    for n, ty, _ in unresolved:
        add_leaf_if_missing(n, ty)

    # Ensure every referenced node has a definition
    for n in list(all_nodes):
        if n not in produced:
            # define any dangling references as leaves
            add_leaf_if_missing(n, produced.get(n, (None, TType((1,)), []))[1] if n in produced else TType((1,)))

    return produced

# ───────────────────────── 5.  Topological order (Kahn) ───────────────────
def topo_sort(graph: Dict[str, tuple[str, TType, List[str]]]) -> List[str]:
    from collections import defaultdict, deque
    indeg = defaultdict(int)
    children = defaultdict(list)
    for node, (_, _, deps) in graph.items():
        for d in deps:
            indeg[node] += 1
            children[d].append(node)
    q = deque([n for n in graph if indeg[n] == 0])
    ordered = []
    while q:
        n = q.popleft()
        ordered.append(n)
        for c in children[n]:
            indeg[c] -= 1
            if indeg[c] == 0:
                q.append(c)
    if len(ordered) != len(graph):
        raise RuntimeError("cycle detected")
    return ordered

# ───────────────────────── 6.  Emit a runnable module source ──────────────
def make_module(
    *,
    depth: int = 8,
    min_chain: int = 3,
    seed: int | None = None,
) -> str:
    if seed is not None:
        random.seed(seed)

    # Random 3-D output (you can vary ranks here if you add more rules)
    out_shape = (
        random.randint(8, 64),
        random.randint(8, 64),
        random.randint(8, 64),
    )

    g = backward_build(out_shape=out_shape, depth=depth, min_chain=min_chain, seed=seed)
    order = topo_sort(g)

    # Emit forward body (no inputs). All leaves are randn; internal nodes are ops.
    lines_body: List[str] = []
    for n in order:
        if n == "t_out":
            continue
        rhs, _, _ = g[n]
        lines_body.append(f"{n} = {rhs}")
    lines_body.append("t_out = " + g["t_out"][0])
    lines_body.append("return t_out")

    # Pretty indentation
    indented = "\n".join(" " * 8 + ln for ln in lines_body)

    return (
        "import torch\n"
        "import torch.nn as nn\n\n"
        "class Model(nn.Module):\n"
        "    def forward(self):\n"
        f"{indented}\n\n"
        "def get_inputs():\n"
        "    return []\n"
    )

# ───────────────────────── 7.  CLI demo ───────────────────────────────────
if __name__ == "__main__":
    for i in range(100):
        # Example: deeper chain with a fixed seed (reproducible)
        print(make_module(depth=10, min_chain=4, seed=i))

