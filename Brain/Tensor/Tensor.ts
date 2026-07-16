// The 2D tensor + autograd node. Every op produces a Tensor that remembers its parents (Prev)
// and how to push gradient to them (BackwardFn); Backward() walks that graph in reverse.
// Data/Grad are flat typed arrays (row-major, index = Row*Cols + Col) in the RUN's precision —
// F64 by default, F32 when Config.Compute.Precision selects it (SetTensorPrecision, wired by
// ActivateFromConfig). F32 halves tape/weight memory and doubles SIMD lanes in the C kernels;
// JS arithmetic still happens in f64 and rounds on store, so op code is precision-agnostic.
// Shape is kept rank-general (number[]) for the ComputeBackend seam, but ops treat it as 2D
// via the Rows/Cols fields.
//
// GRAD IS LAZY: the buffer only materializes on first access (reads of an untouched grad are
// semantically zero either way, so behavior is identical). Eager allocation doubled the memory
// of every forward — including eval/sampling forwards whose grads are NEVER touched — and that
// 2x tape was precisely what capped the training worker pool's width on this machine (16
// concurrent tapes OOM'd). Backward() completes the other half: it releases each interior
// node's buffers as soon as they have been consumed (see ReleaseBuffers).

import { Tape } from "./Tape.ts";

/** The storage type every tensor buffer/kernel signature uses — one of the two float widths. */
export type NumArray = Float64Array | Float32Array;
export type NumArrayCtor = Float64ArrayConstructor | Float32ArrayConstructor;

// The RUN-level storage precision for newly allocated tensor buffers. Module state (not per-tensor)
// on purpose: a run picks ONE precision at activation, before any model is built — per-call
// plumbing through every op signature would buy nothing and cost every call site.
let ActiveCtor: NumArrayCtor = Float64Array;

/** Set the storage precision for all tensors allocated from now on (ActivateFromConfig wires this
 *  from Config.Compute.Precision — call it before constructing a model). */
export function SetTensorPrecision(Precision: "F64" | "F32"): void {
  ActiveCtor = Precision === "F32" ? Float32Array : Float64Array;
}

/** The constructor matching the active precision — for buffers that must match tensor storage
 *  (worker-pool shared slabs, KV caches). */
export function TensorArrayCtor(): NumArrayCtor {
  return ActiveCtor;
}

/** View a (Shared)ArrayBuffer in the active precision. Branches on the concrete constructor because
 *  calling `new` through the UNION constructor type collapses the overloads and rejects
 *  SharedArrayBuffer — each concrete constructor accepts it fine. */
export function NumView(Buffer: ArrayBufferLike, ByteOffset = 0, Length?: number): NumArray {
  if (ActiveCtor === Float32Array) {
    return Length === undefined ? new Float32Array(Buffer, ByteOffset) : new Float32Array(Buffer, ByteOffset, Length);
  }
  return Length === undefined ? new Float64Array(Buffer, ByteOffset) : new Float64Array(Buffer, ByteOffset, Length);
}

function NoBackward(): void {}

const EmptyBuffer = new Float64Array(0);

export class Tensor {
  Data: NumArray;
  Rows: number;
  Cols: number;
  readonly Shape: readonly number[];
  Prev: Tensor[];
  BackwardFn: () => void;
  private GradStore: NumArray | null = null;

  constructor(Rows: number, Cols: number, Data?: NumArray, Prev: Tensor[] = []) {
    this.Rows = Rows;
    this.Cols = Cols;
    this.Shape = [Rows, Cols];
    this.Data = Data ?? new ActiveCtor(Rows * Cols);
    // Only retain the graph when the tape is on (skipped during sampling/eval).
    this.Prev = Tape.On ? Prev : [];
    this.BackwardFn = NoBackward;
  }

  /** Gradient buffer, materialized on first touch (an untouched grad IS zero, so allocating it
   *  eagerly bought nothing and cost a full extra tape of memory). */
  get Grad(): NumArray {
    if (this.GradStore === null) this.GradStore = new ActiveCtor(this.Rows * this.Cols);
    return this.GradStore;
  }

  /** Replace the grad backing store (the training worker pool points params at shared memory). */
  set Grad(Buffer: NumArray) {
    this.GradStore = Buffer;
  }

  /** Number of elements (Rows * Cols) — NOT Data.length, which is 0 after ReleaseBuffers. */
  get Size(): number {
    return this.Rows * this.Cols;
  }

  /** Reset this node's gradient to zero. A never-touched grad is already zero — stays lazy. */
  ZeroGrad(): void {
    this.GradStore?.fill(0);
  }

  /** Drop this node's buffers. Backward() calls it on every INTERIOR node right after its
   *  BackwardFn has fired: reverse-topological order guarantees every consumer of this node ran
   *  earlier, so nothing can read the memory again — holding a full step's tape (Data + Grad for
   *  every activation) until GC noticed it was garbage is what made concurrent training workers
   *  memory-bound. Never called on leaves (params — the optimizer still needs them) or the root
   *  (the caller reads the loss value). A post-release Grad read would silently re-materialize
   *  zeros — acceptable for a consumed interior node, meaningless for a live one, hence the
   *  strict interior-only rule. */
  ReleaseBuffers(): void {
    this.Data = EmptyBuffer;
    this.GradStore = null;
  }
}
