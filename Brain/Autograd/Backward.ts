// The reverse-mode autograd driver — identical for every op, so it is legitimately centralized
// (rule #4: this plumbing must agree with itself everywhere). It topologically orders the graph
// reachable from the root scalar, seeds the root gradient to 1, and fires each node's
// BackwardFn in reverse order. All op-specific gradient math lives inside each op's own
// BackwardFn closure — this driver knows nothing about shapes or ops.

import type { Tensor } from "../Tensor/Tensor.ts";

type BuildFrame = { Node: Tensor; ChildIndex: number };

/** Propagate gradients from a scalar root back to every leaf. Root must be a [1,1] tensor. */
export function Backward(Root: Tensor): void {
  if (Root.Rows !== 1 || Root.Cols !== 1) {
    throw new Error(`Backward: Root must be a scalar [1,1] tensor, got ${Root.Rows}x${Root.Cols}`);
  }

  // Topological order via an explicit-stack post-order DFS (not recursion) so traversal depth is
  // bounded by heap, not the JS call stack — the graph can be as long as the training sequence.
  const Topo: Tensor[] = [];
  const Seen = new Set<Tensor>();
  const Stack: BuildFrame[] = [{ Node: Root, ChildIndex: 0 }];
  Seen.add(Root);

  while (Stack.length > 0) {
    const Frame = Stack[Stack.length - 1];
    if (Frame === undefined) break;
    if (Frame.ChildIndex < Frame.Node.Prev.length) {
      const Parent = Frame.Node.Prev[Frame.ChildIndex];
      Frame.ChildIndex++;
      if (Parent !== undefined && !Seen.has(Parent)) {
        Seen.add(Parent);
        Stack.push({ Node: Parent, ChildIndex: 0 });
      }
    } else {
      Topo.push(Frame.Node);
      Stack.pop();
    }
  }

  Root.Grad.fill(1);
  for (let I = Topo.length - 1; I >= 0; I--) {
    const Node = Topo[I];
    if (Node !== undefined) Node.BackwardFn();
  }
}
