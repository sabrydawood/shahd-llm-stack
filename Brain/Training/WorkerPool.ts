// Sequence-level training parallelism: the batch's sequences fan out across a PERSISTENT pool of
// JS worker threads, each running the UNMODIFIED per-sequence step on its own model instance. This
// is the lever the kernel work could never reach — it parallelizes the serial TS share of the step
// (autograd bookkeeping, elementwise ops, embedding scatter) together with the matmuls, instead of
// only fanning out inside each tiny kernel call (measured: whole-step core use was 1.71 of 28).
//
// Two modes, fixed at pool creation:
//   "pretrain" — fixed-length (Ids, Targets) windows -> ForwardBackward   (used via TrainLoop)
//   "sft"      — variable-length (Ids, LossMask) chats -> SftForwardBackward (used by TrainSftChat)
// Sequence lengths ride a per-worker Lens slab either way, so the SFT batches that used to be
// rejected ("variable-length batches are not pooled") now fan out too — that rejection was worth a
// 6x sequential slowdown on every chat training step.
//
// Sharing model (all zero-copy via SharedArrayBuffer):
//   - WEIGHTS: one shared flat buffer; the MAIN model's param Data arrays are re-pointed into it
//     (values copied once at pool creation), and every worker aliases the same memory. The main
//     thread's optimizer updates weights in place -> workers see them on the next step. Writes are
//     fenced by the step barrier (workers idle while the optimizer runs), so there are no races.
//   - GRADS: one PRIVATE accumulation slab per worker, plus a shared MAIN grad buffer the main
//     model's Grad arrays alias. Reduction is PHASE 2 of the step protocol: every worker owns a
//     disjoint element range and folds ALL workers' slabs into the main buffer in fixed slab order
//     (then scales by 1/BatchSize) — deterministic run-to-run, ~20M serial adds moved off the main
//     thread. No f64 atomics exist; disjoint ranges are what makes lock-free writing sound.
//   - TASKS: per-worker Int32 id/second/lens slabs + one Int32 control word pair, synchronized with
//     Atomics.wait/notify (measured ~4µs round-trip) — no per-step postMessage, no event loop, so
//     accumulation stays SYNCHRONOUS and the training loops keep their sequential structure.
//
// Data order: the main thread alone draws the batch (loader RNG or SFT sampling), so the stream
// consumed is IDENTICAL to the sequential path — resume/reproducibility semantics are unchanged.
//
// Kernel-thread interplay: while workers run, per-call goroutine fan-out inside the Go kernel is
// capped to 1 (SetKernelThreads — process-global, one Go runtime serves all JS threads): the pool
// IS the parallelism. The cap is restored after the barrier so main-thread work (eval) keeps the
// all-cores kernel.

import type { ResolvedConfig } from "../Config/ConfigTypes.ts";
import type { DataLoader } from "../Data/DataLoader.ts";
import type { Shahd } from "../Nn/Shahd.ts";
import type { Tensor } from "../Tensor/Tensor.ts";
import { TensorArrayCtor, NumView } from "../Tensor/Tensor.ts";
import type { TrainingSequence } from "../Sft/ChatTemplate.ts";
import { GetActiveBackend } from "../ComputeBackend/BackendSelector.ts";
import { GoFfiBackend } from "../ComputeBackend/GoFfiBackend.ts";

// Control-word layout (Int32Array): the worker sleeps on CmdSlot, the main thread on StatusSlot.
export const CmdSlot = 0;
export const StatusSlot = 1;
export const SeqCountSlot = 2;
export const CmdCompute = 1;
export const CmdShutdown = 2;
export const CmdReduce = 3;
export const StatusDone = 1;
export const StatusError = 2;

// Loss slab layout (Float64Array[2]).
export const LossOutSlot = 0; // worker -> main: sum of this worker's per-sequence losses
export const InvScaleSlot = 1; // main -> worker: 1/BatchSize, applied during the reduce phase

export type PoolMode = "pretrain" | "sft";

// Backstop so a dead worker thread cannot hang training forever (a healthy sequence takes well
// under a minute even on Micro shapes). Timing out is FATAL — the pool's state is unknown.
const StepWaitMs = 600_000;

/** The one message a worker ever receives: everything it needs, sent once at spawn. */
export type WorkerInit = {
  Config: ResolvedConfig;
  Mode: PoolMode;
  Weights: SharedArrayBuffer; // all params, flat, Parameters() order — aliased by every thread
  GradSlabs: SharedArrayBuffer[]; // EVERY worker's private accumulator (this worker writes only its own)
  MyIndex: number; // which slab above is mine
  MainGrad: SharedArrayBuffer; // the main model's Grad backing — reduce-phase output
  ReduceFrom: number; // my disjoint element range in the flat param space
  ReduceTo: number;
  Ctl: SharedArrayBuffer; // Int32 control words (slots above)
  Ids: SharedArrayBuffer; // Int32 [MaxSeqs * MaxLen] token ids
  Second: SharedArrayBuffer; // Int32 [MaxSeqs * MaxLen]: pretrain -> targets, sft -> loss mask (0/1)
  Lens: SharedArrayBuffer; // Int32 [MaxSeqs] — actual length of each sequence this step
  Loss: SharedArrayBuffer; // Float64 [2] — see the slot constants
  MaxSeqs: number;
  MaxLen: number;
};

/** Re-point a model's parameter Data (and optionally Grad) into flat shared memory, preserving
 *  values when asked. Layout = Parameters() order, which both sides build from the same Config.
 *  The element width follows the run's storage precision (main thread and workers both activate
 *  from the SAME Config before aliasing, so the two sides always view the buffers identically —
 *  an F32 run shares Float32 slabs, halving pool memory and feeding the f32 kernels directly). */
export function AliasParams(Params: Tensor[], Weights: SharedArrayBuffer, Grads: SharedArrayBuffer | null, CopyValues: boolean): void {
  const Bytes = TensorArrayCtor().BYTES_PER_ELEMENT;
  let Offset = 0;
  for (const P of Params) {
    const View = NumView(Weights, Offset * Bytes, P.Size);
    if (CopyValues) View.set(P.Data as Float64Array); // set() converts across widths; the cast picks one union arm
    P.Data = View;
    if (Grads !== null) P.Grad = NumView(Grads, Offset * Bytes, P.Size);
    Offset += P.Size;
  }
  if (Offset * Bytes !== Weights.byteLength) {
    throw new Error(`WorkerPool: param layout mismatch — ${Offset * Bytes} bytes of params vs ${Weights.byteLength} shared`);
  }
}

export class TrainWorkerPool {
  private Workers: Worker[] = [];
  private Ctls: Int32Array[] = [];
  private IdsViews: Int32Array[] = [];
  private SecondViews: Int32Array[] = [];
  private LensViews: Int32Array[] = [];
  private LossViews: Float64Array[] = [];
  private Mode: PoolMode;
  private MaxSeqs: number;
  private MaxLen: number;

  /** Use CreateTrainWorkerPool — the constructor only wires already-initialized pieces. */
  constructor(Mode: PoolMode, MaxSeqs: number, MaxLen: number) {
    this.Mode = Mode;
    this.MaxSeqs = MaxSeqs;
    this.MaxLen = MaxLen;
  }

  /** @internal registration used by CreateTrainWorkerPool for each spawned worker. */
  AddWorker(W: Worker, Ctl: Int32Array, Ids: Int32Array, Second: Int32Array, Lens: Int32Array, Loss: Float64Array): void {
    this.Workers.push(W);
    this.Ctls.push(Ctl);
    this.IdsViews.push(Ids);
    this.SecondViews.push(Second);
    this.LensViews.push(Lens);
    this.LossViews.push(Loss);
  }

  get WorkerCount(): number {
    return this.Workers.length;
  }

  /** Drop-in parallel AccumulateGradients for PRETRAIN: BatchSize fixed-length sequences fanned
   *  across the pool, gradients reduced into the main model's Grad buffers (scaled 1/BatchSize),
   *  mean loss returned. Fully synchronous — the caller blocks on the step barrier. */
  Accumulate(Loader: DataLoader, BatchSize: number): number {
    if (this.Mode !== "pretrain") throw new Error("WorkerPool.Accumulate: this pool was created for SFT — use AccumulateSft");
    const W = this.WorkerCount;
    this.CheckCapacity(BatchSize);
    const Counts = new Array<number>(W).fill(0);
    // The main thread ALONE consumes the loader (identical RNG stream to the sequential path),
    // round-robin so worker loads differ by at most one sequence.
    for (let B = 0; B < BatchSize; B++) {
      const { Ids, Targets } = Loader.GetSequence();
      if (Ids.length !== this.MaxLen || Targets.length !== this.MaxLen) {
        throw new Error(`WorkerPool: pretrain sequence length ${Ids.length} != BlockSize ${this.MaxLen}`);
      }
      const Wi = B % W;
      this.IdsViews[Wi].set(Ids, Counts[Wi] * this.MaxLen);
      this.SecondViews[Wi].set(Targets, Counts[Wi] * this.MaxLen);
      this.LensViews[Wi][Counts[Wi]] = Ids.length;
      Counts[Wi]++;
    }
    return this.RunStep(Counts, BatchSize);
  }

  /** Parallel SFT accumulation: the caller samples the batch (preserving its RNG order) and hands
   *  the rendered sequences over; lengths may vary up to BlockSize+1. Same reduction/scaling/mean
   *  semantics as the sequential TrainSftChat loop. */
  AccumulateSft(Sequences: TrainingSequence[]): number {
    if (this.Mode !== "sft") throw new Error("WorkerPool.AccumulateSft: this pool was created for pretrain — use Accumulate");
    const W = this.WorkerCount;
    this.CheckCapacity(Sequences.length);
    const Counts = new Array<number>(W).fill(0);
    for (let B = 0; B < Sequences.length; B++) {
      const Seq = Sequences[B];
      const Len = Seq.Ids.length;
      if (Len > this.MaxLen) throw new Error(`WorkerPool: SFT sequence length ${Len} exceeds the pool's MaxLen ${this.MaxLen}`);
      if (Seq.LossMask.length !== Len) throw new Error(`WorkerPool: SFT LossMask length ${Seq.LossMask.length} != Ids length ${Len}`);
      const Wi = B % W;
      const Base = Counts[Wi] * this.MaxLen;
      const IdsView = this.IdsViews[Wi];
      const MaskView = this.SecondViews[Wi];
      for (let I = 0; I < Len; I++) {
        IdsView[Base + I] = Seq.Ids[I];
        MaskView[Base + I] = Seq.LossMask[I] ? 1 : 0;
      }
      this.LensViews[Wi][Counts[Wi]] = Len;
      Counts[Wi]++;
    }
    return this.RunStep(Counts, Sequences.length);
  }

  private CheckCapacity(BatchSize: number): void {
    const Needed = Math.ceil(BatchSize / this.WorkerCount);
    if (Needed > this.MaxSeqs) {
      throw new Error(`WorkerPool: batch of ${BatchSize} needs ${Needed} seqs/worker but the pool was sized for ${this.MaxSeqs}`);
    }
  }

  /** Phase 1 (compute) + phase 2 (parallel reduce), each behind an Atomics barrier. */
  private RunStep(Counts: number[], BatchSize: number): number {
    const W = this.WorkerCount;
    // While workers run, kernel calls must not fan out goroutines (the pool is the parallelism).
    const Backend = GetActiveBackend();
    const Kernel = Backend instanceof GoFfiBackend ? Backend : null;
    Kernel?.SetKernelThreads?.(1);
    try {
      for (let Wi = 0; Wi < W; Wi++) {
        Atomics.store(this.Ctls[Wi], SeqCountSlot, Counts[Wi]);
        this.Dispatch(Wi, CmdCompute);
      }
      this.Await("compute");

      const Inv = 1 / BatchSize;
      for (let Wi = 0; Wi < W; Wi++) {
        this.LossViews[Wi][InvScaleSlot] = Inv;
        this.Dispatch(Wi, CmdReduce);
      }
      this.Await("reduce");

      let TotalLoss = 0;
      for (let Wi = 0; Wi < W; Wi++) TotalLoss += this.LossViews[Wi][LossOutSlot];
      return TotalLoss * Inv;
    } finally {
      Kernel?.SetKernelThreads?.(0);
    }
  }

  private Dispatch(Wi: number, Cmd: number): void {
    const Ctl = this.Ctls[Wi];
    Atomics.store(Ctl, StatusSlot, 0);
    Atomics.store(Ctl, CmdSlot, Cmd);
    Atomics.notify(Ctl, CmdSlot);
  }

  private Await(Phase: string): void {
    for (let Wi = 0; Wi < this.Ctls.length; Wi++) {
      const Ctl = this.Ctls[Wi];
      while (Atomics.load(Ctl, StatusSlot) === 0) {
        if (Atomics.wait(Ctl, StatusSlot, 0, StepWaitMs) === "timed-out") {
          throw new Error(`WorkerPool: worker ${Wi} did not finish the ${Phase} phase within ${StepWaitMs}ms — pool state unknown, aborting`);
        }
      }
      if (Atomics.load(Ctl, StatusSlot) === StatusError) {
        throw new Error(`WorkerPool: worker ${Wi} failed during the ${Phase} phase — see its stderr for the cause`);
      }
    }
  }

  /** Shut the workers down. The pool is unusable afterwards. */
  Dispose(): void {
    for (let Wi = 0; Wi < this.Workers.length; Wi++) {
      Atomics.store(this.Ctls[Wi], CmdSlot, CmdShutdown);
      Atomics.notify(this.Ctls[Wi], CmdSlot);
      this.Workers[Wi].terminate();
    }
    this.Workers = [];
    this.Ctls = [];
  }
}

/** Build a pool of Config.Training.Workers threads (capped by BatchSize) around Model: swaps the
 *  model's parameters AND grads into shared memory, spawns the workers, and waits for each to
 *  alias the weights and report ready. The model trains EXACTLY as before from the caller's point
 *  of view — only where the per-sequence work physically runs changes. */
export async function CreateTrainWorkerPool(Model: Shahd, Config: ResolvedConfig, Mode: PoolMode = "pretrain"): Promise<TrainWorkerPool> {
  const Requested = Config.Training.Workers;
  if (Requested < 1) throw new Error("CreateTrainWorkerPool: Config.Training.Workers must be >= 1");
  const BatchSize = Config.Training.BatchSize;
  const W = Math.min(Requested, BatchSize);

  const Params = Model.Parameters();
  let Total = 0;
  for (const P of Params) Total += P.Size;
  const SlabBytes = Total * TensorArrayCtor().BYTES_PER_ELEMENT; // width follows the run's precision
  const Weights = new SharedArrayBuffer(SlabBytes);
  const MainGrad = new SharedArrayBuffer(SlabBytes);
  // The main model's grads alias MainGrad so the workers' reduce phase can write them directly;
  // the reduce ASSIGNS every element (ranges cover the whole space), so no zeroing is ever needed.
  AliasParams(Params, Weights, MainGrad, true);

  const GradSlabs: SharedArrayBuffer[] = [];
  for (let Wi = 0; Wi < W; Wi++) GradSlabs.push(new SharedArrayBuffer(SlabBytes));

  const MaxSeqs = Math.ceil(BatchSize / W);
  // SFT sequences carry one extra token (inputs = Ids[:-1], targets = Ids[1:] happen inside the
  // step), so a rendered chat may legitimately be BlockSize + 1 ids long.
  const MaxLen = Mode === "sft" ? Config.Model.BlockSize + 1 : Config.Model.BlockSize;
  const Pool = new TrainWorkerPool(Mode, MaxSeqs, MaxLen);

  const PerRange = Math.ceil(Total / W);
  const Ready: Promise<void>[] = [];
  for (let Wi = 0; Wi < W; Wi++) {
    const Ctl = new SharedArrayBuffer(4 * 4);
    const Ids = new SharedArrayBuffer(MaxSeqs * MaxLen * 4);
    const Second = new SharedArrayBuffer(MaxSeqs * MaxLen * 4);
    const Lens = new SharedArrayBuffer(MaxSeqs * 4);
    const Loss = new SharedArrayBuffer(2 * 8);
    const Init: WorkerInit = {
      Config,
      Mode,
      Weights,
      GradSlabs,
      MyIndex: Wi,
      MainGrad,
      ReduceFrom: Math.min(Wi * PerRange, Total),
      ReduceTo: Math.min((Wi + 1) * PerRange, Total),
      Ctl,
      Ids,
      Second,
      Lens,
      Loss,
      MaxSeqs,
      MaxLen,
    };

    const WorkerRef = new Worker(new URL("./TrainWorker.ts", import.meta.url));
    Ready.push(
      new Promise<void>((Resolve, Reject) => {
        WorkerRef.onmessage = () => Resolve();
        WorkerRef.onerror = (Event) => Reject(new Error(`WorkerPool: worker ${Wi} failed to start: ${Event.message}`));
      }),
    );
    WorkerRef.postMessage(Init);
    Pool.AddWorker(WorkerRef, new Int32Array(Ctl), new Int32Array(Ids), new Int32Array(Second), new Int32Array(Lens), new Float64Array(Loss));
  }
  await Promise.all(Ready);
  return Pool;
}
