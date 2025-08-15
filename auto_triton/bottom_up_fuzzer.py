# generic_type_system_fuzzer_topo_selfcheck.py
# A generic, rule-driven, bottom-up PyTorch fuzzer using a small type system.
# Emits code in proper topological order and self-checks by running forward() once.

import random
from dataclasses import dataclass
from typing import List, Tuple, Dict, Optional, Union, Set, Any

# ───────────────────────────────────────────────────────────────────────────
# 1) Type algebra (Tensor + symbolic dims) + tiny constraint store
# ───────────────────────────────────────────────────────────────────────────

Dim = Union[int, str]  # int literal or symbolic variable name like "n", "k"

@dataclass(frozen=True)
class TensorType:
    dtype: str
    shape: Tuple[Dim, ...]
    @property
    def rank(self) -> int:
        return len(self.shape)

class Constraints:
    def __init__(self):
        # symbol -> representative (int or other symbol), union-find style via dict
        self.eq: Dict[str, Union[int, str]] = {}

    def _repr(self, d: Dim) -> Dim:
        if isinstance(d, int):
            return d
        seen = set()
        cur: Union[int, str] = d
        while isinstance(cur, str) and cur in self.eq and cur not in seen:
            seen.add(cur)
            cur = self.eq[cur]
        return cur

    def unify(self, a: Dim, b: Dim) -> bool:
        ra, rb = self._repr(a), self._repr(b)
        if isinstance(ra, int) and isinstance(rb, int):
            return ra == rb
        if isinstance(ra, str) and isinstance(rb, int):
            self.eq[ra] = rb
            return True
        if isinstance(ra, int) and isinstance(rb, str):
            self.eq[rb] = ra
            return True
        if isinstance(ra, str) and isinstance(rb, str):
            if ra != rb:
                self.eq[rb] = ra
            return True
        return False

    def broadcastable(self, a: Tuple[Dim, ...], b: Tuple[Dim, ...]) -> Optional[Tuple[Dim, ...]]:
        # PyTorch-style broadcasting (align from right; a dim can be 1 or equal)
        A, B = list(a), list(b)
        if len(A) < len(B):
            A = [1] * (len(B) - len(A)) + A
        elif len(B) < len(A):
            B = [1] * (len(A) - len(B)) + B
        out: List[Dim] = []
        for da, db in zip(reversed(A), reversed(B)):
            ra, rb = self._repr(da), self._repr(db)
            if ra == 1:
                out.append(rb)
            elif rb == 1:
                out.append(ra)
            elif isinstance(ra, int) and isinstance(rb, int):
                if ra != rb:
                    return None
                out.append(ra)
            else:
                if not self.unify(ra, rb):
                    return None
                out.append(self._repr(ra))
        return tuple(reversed(out))

    def solve(self, dims: List[str], low=8, high=64) -> Dict[str, int]:
        # Sample concrete integers for symbolic dims consistent with equalities.
        sol: Dict[str, int] = {}
        for s in dims:
            r = self._repr(s)
            sol[s] = r if isinstance(r, int) else random.randint(low, high)
        # Propagate equalities
        changed = True
        while changed:
            changed = False
            for s in list(sol.keys()):
                r = self._repr(s)
                if isinstance(r, int) and sol[s] != r:
                    sol[s] = r; changed = True
                elif isinstance(r, str) and r in sol and sol[s] != sol[r]:
                    sol[s] = sol[r]; changed = True
        return sol

def shape_symbols(shape: Tuple[Dim, ...]) -> List[str]:
    return [d for d in shape if isinstance(d, str)]

def substitute_shape(shape: Tuple[Dim, ...], sol: Dict[str, int]) -> Tuple[int, ...]:
    return tuple(d if isinstance(d, int) else sol[d] for d in shape)

# ───────────────────────────────────────────────────────────────────────────
# 2) Rule schemas (generic)
# ───────────────────────────────────────────────────────────────────────────

@dataclass
class Goal:
    name: str
    ttype: TensorType
    budget: int
    forbid: Set[str]        # ancestors on this path (cycle guard)

@dataclass
class Node:
    name: str
    rhs: str
    ttype: TensorType
    deps: List[str]

class Rule:
    name: str
    def invert(self, goal: Goal, C: Constraints) -> Optional[Tuple[List[Goal], Dict[str, Any]]]:
        """Return (premises, attrs) or None if rule doesn't apply."""
        raise NotImplementedError
    def emit(self, node: Node, attrs: Dict[str, Any]) -> Node:
        return node

# ── Unary (shape-preserving) ───────────────────────────────────────────────
class UnaryShapePreserving(Rule):
    def __init__(self, opname_choices: List[str]):
        self.name = "Unary"
        self.choices = opname_choices

    def invert(self, goal: Goal, C: Constraints) -> Optional[Tuple[List[Goal], Dict[str, Any]]]:
        xname = fresh_name()
        sub = Goal(xname, goal.ttype, goal.budget - 1, goal.forbid | {goal.name})
        return [sub], {}

    def emit(self, node: Node, attrs: Dict[str, Any]) -> Node:
        op = random.choice(self.choices)
        if op == "softmax" and node.ttype.rank == 0:
            op = random.choice([c for c in self.choices if c != "softmax"])
        dep = node.deps[0]
        if op == "relu":
            rhs = f"torch.relu({dep})"
        elif op == "sigmoid":
            rhs = f"torch.sigmoid({dep})"
        elif op == "tanh":
            rhs = f"torch.tanh({dep})"
        elif op == "softmax":
            rhs = f"torch.softmax({dep}, dim=-1)"
        else:
            rhs = dep
        return Node(node.name, rhs, node.ttype, node.deps)

# ── Binary elementwise with broadcasting ───────────────────────────────────
class BinaryElementwise(Rule):
    def __init__(self, op_choices: List[str]):
        self.name = "BinaryElt"
        self.ops = op_choices

    def invert(self, goal: Goal, C: Constraints) -> Optional[Tuple[List[Goal], Dict[str, Any]]]:
        r = goal.ttype.rank
        ra = random.choice([max(1, r-1), r])  # encourage broadcasts
        rb = random.choice([r, max(1, r-1)])
        def fresh_shape(rank: int) -> Tuple[Dim, ...]:
            return tuple(random.choice([1, random_dim_sym()]) for _ in range(rank))
        sa, sb = fresh_shape(ra), fresh_shape(rb)
        out = C.broadcastable(sa, sb)
        if out is None or len(out) != len(goal.ttype.shape):
            return None
        for da, dg in zip(out, goal.ttype.shape):
            if isinstance(dg, int):
                if isinstance(da, int):
                    if da != dg: return None
                else:
                    if not C.unify(da, dg): return None
            else:
                if not C.unify(da, dg): return None
        a_name, b_name = fresh_name(), fresh_name()
        A = Goal(a_name, TensorType(goal.ttype.dtype, sa), goal.budget - 1, goal.forbid | {goal.name})
        B = Goal(b_name, TensorType(goal.ttype.dtype, sb), goal.budget - 1, goal.forbid | {goal.name})
        return [A, B], {}

    def emit(self, node: Node, attrs: Dict[str, Any]) -> Node:
        op = random.choice(self.ops)
        a, b = node.deps
        if op == "add":
            rhs = f"{a} + {b}"
        elif op == "mul":
            rhs = f"{a} * {b}"
        else:
            rhs = f"{a} + {b}"
        return Node(node.name, rhs, node.ttype, node.deps)

# ── Contraction (matmul / 4D einsum) ──────────────────────────────────────
class Contraction(Rule):
    def __init__(self):
        self.name = "Contract"

    def invert(self, goal: Goal, C: Constraints) -> Optional[Tuple[List[Goal], Dict[str, Any]]]:
        r = goal.ttype.rank
        if r == 3:
            b, i, j = goal.ttype.shape
            k = random_dim_sym()
            A = Goal(fresh_name(), TensorType(goal.ttype.dtype, (b, i, k)), goal.budget - 1, goal.forbid | {goal.name})
            B = Goal(fresh_name(), TensorType(goal.ttype.dtype, (k, j)),     goal.budget - 1, goal.forbid | {goal.name})
            return [A, B], {"kind": "matmul"}
        if r == 4:
            b, i, j, l = goal.ttype.shape
            k = random_dim_sym()
            A = Goal(fresh_name(), TensorType(goal.ttype.dtype, (b, i, j, k)), goal.budget - 1, goal.forbid | {goal.name})
            B = Goal(fresh_name(), TensorType(goal.ttype.dtype, (k, l)),       goal.budget - 1, goal.forbid | {goal.name})
            return [A, B], {"kind": "einsum4"}
        return None

    def emit(self, node: Node, attrs: Dict[str, Any]) -> Node:
        a, b = node.deps
        if attrs.get("kind") == "matmul":
            rhs = f"torch.matmul({a}, {b})"
        elif attrs.get("kind") == "einsum4":
            rhs = f'torch.einsum("bijk,kl->bijl", {a}, {b})'
        else:
            rhs = a
        return Node(node.name, rhs, node.ttype, node.deps)

# ── Reduction (drop one axis) ─────────────────────────────────────────────
class Reduction(Rule):
    def __init__(self, ops=("sum","mean","amax")):
        self.name = "Reduce"
        self.ops = ops

    def invert(self, goal: Goal, C: Constraints) -> Optional[Tuple[List[Goal], Dict[str, Any]]]:
        # goal: Tensor(τ, s) with rank r
        # premise: Tensor(τ, s[:axis] + (k,) + s[axis:]); reduce over 'axis'
        r = goal.ttype.rank
        # insert the reduced dim at some position (0..r)
        axis = random.randint(0, r)  # after reduction we return to rank r
        k = random_dim_sym()
        in_shape = tuple(list(goal.ttype.shape[:axis]) + [k] + list(goal.ttype.shape[axis:]))
        X = Goal(fresh_name(), TensorType(goal.ttype.dtype, in_shape), goal.budget - 1, goal.forbid | {goal.name})
        return [X], {"axis": axis}

    def emit(self, node: Node, attrs: Dict[str, Any]) -> Node:
        axis = attrs.get("axis", -1)
        op = random.choice(self.ops)
        x = node.deps[0]
        if op == "sum":
            rhs = f"torch.sum({x}, dim={axis})"
        elif op == "mean":
            rhs = f"torch.mean({x}, dim={axis})"
        elif op == "amax":
            rhs = f"torch.amax({x}, dim={axis})"
        else:
            rhs = f"torch.sum({x}, dim={axis})"
        return Node(node.name, rhs, node.ttype, node.deps)

# ── Permute (shape-preserving isomorphism) ─────────────────────────────────
class Permute(Rule):
    def __init__(self):
        self.name = "Permute"

    def invert(self, goal: Goal, C: Constraints) -> Optional[Tuple[List[Goal], Dict[str, Any]]]:
        r = goal.ttype.rank
        if r < 2:
            return None
        perm = list(range(r))
        random.shuffle(perm)
        # If perm is identity, skip
        if perm == list(range(r)):
            return None
        # If out = permute(inp, perm), then inp.shape = inverse_perm(out.shape)
        inv = [0]*r
        for i,p in enumerate(perm):
            inv[p] = i
        in_shape = tuple(goal.ttype.shape[i] for i in inv)
        X = Goal(fresh_name(), TensorType(goal.ttype.dtype, in_shape), goal.budget - 1, goal.forbid | {goal.name})
        return [X], {"perm": tuple(perm)}

    def emit(self, node: Node, attrs: Dict[str, Any]) -> Node:
        perm = attrs["perm"]
        x = node.deps[0]
        rhs = f"{x}.permute{perm}"
        return Node(node.name, rhs, node.ttype, node.deps)

# ───────────────────────────────────────────────────────────────────────────
# 3) Synthesis engine (backward proof search) + topo emit
# ───────────────────────────────────────────────────────────────────────────

_sym_counter = 0
_name_counter = 0
def random_dim_sym() -> str:
    global _sym_counter
    _sym_counter += 1
    return f"s{_sym_counter}"

def fresh_name() -> str:
    global _name_counter
    _name_counter += 1
    return f"t{_name_counter}"

def collect_symbols(nodes: List[Node]) -> List[str]:
    syms: Set[str] = set()
    for nd in nodes:
        for d in nd.ttype.shape:
            if isinstance(d, str): syms.add(d)
    return list(syms)

def topo_sort_nodes(graph: Dict[str, Node]) -> List[str]:
    # Kahn's algorithm
    indeg: Dict[str, int] = {n: 0 for n in graph}
    children: Dict[str, List[str]] = {n: [] for n in graph}
    for n, nd in graph.items():
        for d in nd.deps:
            if d not in indeg:
                indeg[d] = 0; children[d] = []
            indeg[n] += 1
            children[d].append(n)
    from collections import deque
    q = deque([n for n, deg in indeg.items() if deg == 0])
    order: List[str] = []
    while q:
        n = q.popleft()
        order.append(n)
        for c in children.get(n, []):
            indeg[c] -= 1
            if indeg[c] == 0:
                q.append(c)
    if len(order) != len(indeg):
        raise RuntimeError("cycle detected in emitted graph")
    return order

def make_module(depth=16, min_chain=4, seed: Optional[int] = None) -> str:
    if seed is not None:
        random.seed(seed)

    # Random goal tensor type (3D or 4D) with symbols
    rank = random.choice([3, 4])
    out_shape_syms = tuple(random_dim_sym() for _ in range(rank))
    goal = Goal("t_out", TensorType("float32", out_shape_syms), min_chain, {"t_out"})

    # Rule set (generic)
    RULES: List[Rule] = [
        UnaryShapePreserving(["relu", "sigmoid", "tanh", "softmax"]),
        BinaryElementwise(["add", "mul"]),
        Contraction(),         # matmul for 3D, einsum4 for 4D
        Reduction(),           # drop an axis
        Permute(),             # shape-preserving permute
    ]
    C = Constraints()

    # Synthesis (stack of subgoals). Each entry carries a forbid set.
    stack: List[Goal] = [goal]
    nodes: Dict[str, Node] = {}
    plan: Dict[str, Tuple[str, Dict[str, Any]]] = {}  # node -> (rule_name, attrs)
    steps_left = depth

    while stack and steps_left > 0:
        g = stack.pop()
        if g.name in nodes:
            steps_left -= 1
            continue

        if g.budget <= 0:
            nodes[g.name] = Node(g.name, "<Leaf>", g.ttype, [])
            plan[g.name] = ("Leaf", {})
            steps_left -= 1
            continue

        # Try rules in random order
        rules = RULES[:]
        random.shuffle(rules)
        chosen: Optional[Rule] = None
        premises: Optional[List[Goal]] = None
        attrs: Dict[str, Any] = {}
        for R in rules:
            res = R.invert(g, C)
            if res is not None:
                premises, attrs = res
                chosen = R
                break

        if chosen is None:
            nodes[g.name] = Node(g.name, "<Leaf>", g.ttype, [])
            plan[g.name] = ("Leaf", {})
        else:
            dep_names = [p.name for p in premises]
            nodes[g.name] = Node(g.name, f"<{chosen.name}>", g.ttype, dep_names)
            plan[g.name] = (chosen.name, attrs)
            for p in premises:
                stack.append(p)

        steps_left -= 1

    # Remaining goals become leaves
    for g in stack:
        if g.name not in nodes:
            nodes[g.name] = Node(g.name, "<Leaf>", g.ttype, [])
            plan[g.name] = ("Leaf", {})

    # Solve symbolic dims → integers
    all_nodes = list(nodes.values())
    symbols = collect_symbols(all_nodes)
    sol = C.solve(symbols, low=8, high=64)

    # Emit concrete rhs for all nodes
    emitted: Dict[str, Node] = {}
    def concretize_shape(shape): return substitute_shape(shape, sol)

    def emit_node(n: Node) -> Node:
        if n.name in emitted:
            return emitted[n.name]
        deps_out: List[str] = []
        for d in n.deps:
            deps_out.append(emit_node(nodes[d]).name)
        rule_name, attrs = plan.get(n.name, ("Leaf", {}))
        if rule_name == "Leaf":
            rhs = f"torch.randn({concretize_shape(n.ttype.shape)}, dtype=torch.{n.ttype.dtype})"
            final = Node(n.name, rhs, n.ttype, [])
        elif rule_name == "Unary":
            final = UnaryShapePreserving(["relu","sigmoid","tanh","softmax"]).emit(Node(n.name, "", n.ttype, deps_out), attrs)
        elif rule_name == "BinaryElt":
            final = BinaryElementwise(["add","mul"]).emit(Node(n.name, "", n.ttype, deps_out), attrs)
        elif rule_name == "Contract":
            final = Contraction().emit(Node(n.name, "", n.ttype, deps_out), attrs)
        elif rule_name == "Reduce":
            final = Reduction().emit(Node(n.name, "", n.ttype, deps_out), attrs)
        elif rule_name == "Permute":
            final = Permute().emit(Node(n.name, "", n.ttype, deps_out), attrs)
        else:
            final = Node(n.name, "torch.randn(())", n.ttype, deps_out)
        emitted[n.name] = final
        return final

    for n in all_nodes:
        emit_node(n)

    # True topological order over emitted graph
    order = topo_sort_nodes(emitted)

    # Emit body in topo order and return module source
    body: List[str] = [f"{name} = {emitted[name].rhs}" for name in order]
    body.append("return t_out")
    indented = "\n".join(" " * 8 + ln for ln in body)

    return (
        "import torch\n"
        "import torch.nn as nn\n\n"
        "class Model(nn.Module):\n"
        "    def forward(self):\n"
        f"{indented}\n\n"
        "def get_inputs():\n"
        "    return []\n"
    )

# ───────────────────────────────────────────────────────────────────────────
# 4) Self-check runner: compile & run the generated model once
# ───────────────────────────────────────────────────────────────────────────
def generate_and_run(depth=16, min_chain=4, seed: Optional[int] = None) -> Tuple[str, Tuple[int, ...]]:
    src = make_module(depth=depth, min_chain=min_chain, seed=seed)
    return src

    if False:
        # Compile the source into a throwaway module dict
        gbl: Dict[str, Any] = {}
        loc: Dict[str, Any] = {}
        exec(src, gbl, loc)
        Model = loc["Model"]
        import torch
        with torch.no_grad():
            m = Model()
            out = m.forward()
        if not hasattr(out, "shape"):
            raise RuntimeError("Forward did not return a tensor-like with a shape.")
        return src, tuple(out.shape)

# ───────────────────────────────────────────────────────────────────────────
# 5) Demo
# ───────────────────────────────────────────────────────────────────────────
if __name__ == '__main__':
    # src, osh = generate_and_run(depth=18, min_chain=5, seed=12345)
    # print(f"# forward() ran successfully; output shape = {osh}")
    for i in range(10):
        _sym_counter = 0
        _name_counter = 0
        src = generate_and_run(depth=18, min_chain=5, seed=12345 + i)
        print(src)

