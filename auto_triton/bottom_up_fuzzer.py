import random, time
from typing import Tuple, List, Dict, Optional, NamedTuple

# ───────────────────────── 1.  TensorType ──────────
class TType(NamedTuple):
    shape: Tuple[int, ...]
    dtype: str = "float32"
    @property
    def rank(self): return len(self.shape)

# ───────────────────────── 2.  Op registry (fwd str + deps) ─
UNARY_FMTS = [
    ("relu",    lambda x: f"torch.relu({x})"),
    ("sigmoid", lambda x: f"torch.sigmoid({x})"),
    ("tanh",    lambda x: f"torch.tanh({x})"),
    ("softmax", lambda x: f"torch.softmax({x}, dim=-1)")
]

# helpers to create new var names
def fresh(counter: List[int]):
    counter[0] += 1
    return f"t{counter[0]}"

# ───────────────────────── 3.  Bottom-up builder ───
def backward_build(depth:int,
                   X,N,M,K,L) -> Dict[str, Tuple[str,TType,List[str]]]:
    """
    Returns dict  name -> (rhs_string, TType, deps_list)
    where deps_list are *child* tensor names the rhs refers to.
    """
    leaves = {
        "t0": TType((X,N,M,K)),
        "t1": TType((N,M,K)),
        "t2": TType((K,L)),
    }

    produced: Dict[str, Tuple[str,TType,List[str]]] = {}
    counter=[2]          # t0, t1, t2 reserved
    unresolved=[("t_out", TType((N,M,L)))]

    while unresolved and depth:
        name, ty = unresolved.pop()
        # pick op type compatible with target
        choice = random.choice(
            ["unary","matmul","einsum"] if ty.rank>=3 else ["unary"]
        )

        if choice=="unary":
            op,fmt = random.choice(UNARY_FMTS)
            src=fresh(counter)
            produced[src]=(f"torch.randn({ty.shape}, dtype=torch.float32)", ty, [])
            produced[name]=(fmt(src), ty, [src])
            unresolved.append((src,ty))

        elif choice=="matmul" and ty.rank==3:
            B="t2"
            Bty=leaves[B]
            A=fresh(counter)
            Aty=TType((ty.shape[0],ty.shape[1],Bty.shape[0]))
            produced[A]=(f"torch.randn({Aty.shape}, dtype=torch.float32)",Aty,[])
            produced[name]=(f"torch.matmul({A}, {B})", ty, [A,B])
            unresolved.append((A,Aty))

        elif choice=="einsum" and ty.rank==3:
            B="t2"; Bty=leaves[B]
            A=fresh(counter)
            Aty=TType((ty.shape[0],ty.shape[1],Bty.shape[0],Bty.shape[0]))
            produced[A]=(f"torch.randn({Aty.shape}, dtype=torch.float32)",Aty,[])
            produced[name]=(f'torch.einsum("bijk,kl->bijl", {A}, {B})', ty, [A,B])
            unresolved.append((A,Aty))
        depth-=1

    # add leaves
    for n,ty in leaves.items():
        produced.setdefault(n,(f"torch.randn({ty.shape}, dtype=torch.float32)",ty,[]))

    return produced

# ───────────────────────── 4.  Topological order (Kahn) ─────
def topo_sort(graph: Dict[str, Tuple[str,TType,List[str]]]) -> List[str]:
    from collections import defaultdict, deque
    indeg=defaultdict(int)
    children=defaultdict(list)
    for node,(rhs,_,deps) in graph.items():
        for d in deps:
            indeg[node]+=1
            children[d].append(node)
    q=deque([n for n in graph if indeg[n]==0])
    ordered=[]
    while q:
        n=q.popleft()
        ordered.append(n)
        for c in children[n]:
            indeg[c]-=1
            if indeg[c]==0:
                q.append(c)
    if len(ordered)!=len(graph):
        raise RuntimeError("cycle detected")
    return ordered

# ───────────────────────── 5.  Emit module source ───────────
def make_module(depth=8)->str:
    X,N,M,K,L=[random.randint(16,128) for _ in range(5)]
    g=backward_build(depth,X,N,M,K,L)
    order=topo_sort(g)

    inputs  = [n for n in ("t0","t1","t2")]
    lines_inputs=[f"{n} = {g[n][0]}" for n in inputs]
    lines_body  =[f"{n} = {g[n][0]}" for n in order if n not in inputs+["t_out"]]
    lines_body.append("t_out = "+g["t_out"][0])
    lines_body.append("return t_out")

    indent=lambda s,sp: "\n".join(" "*sp+ln for ln in s.splitlines())

    return f"""
import torch, torch.nn as nn
class Model(nn.Module):
    def forward(self, {', '.join(inputs)}):
{indent('\n'.join(lines_body),8)}

def get_inputs():
{indent('\n'.join(lines_inputs),4)}
    return [{', '.join(inputs)}]
"""

# ───────────────────────── 6.  demo ─────────────────────────
if __name__=="__main__":
    random.seed(time.time_ns())
    print(make_module(depth=8))

