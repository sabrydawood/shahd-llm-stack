// The reverse-mode autograd driver — identical for every op, so it is legitimately centralized
// (rule #4: this plumbing must agree with itself everywhere). It topologically orders the graph
// reachable from the root scalar, seeds the root gradient to 1, and fires each node's
// BackwardFn in reverse order. All op-specific gradient math lives inside each op's own
// BackwardFn closure — this driver knows nothing about shapes or ops.

import type { Tensor } from "../Tensor/Tensor.ts";

/** Propagate gradients from a scalar root back to every leaf. Root must be a [1,1] tensor. */
export function Backward(Root: Tensor): void {
  const Topo: Tensor[] = [];
  const Seen = new Set<Tensor>();

  const Build = (Node: Tensor): void => {
    if (Seen.has(Node)) return;
    Seen.add(Node);
    for (const Parent of Node.Prev) Build(Parent);
    Topo.push(Node);
  };
  Build(Root);

  Root.Grad.fill(1);
  for (let I = Topo.length - 1; I >= 0; I--) {
    const Node = Topo[I];
    if (Node !== undefined) Node.BackwardFn();
  }
}
