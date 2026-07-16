// Worker-thread entry for the training pool (see WorkerPool.ts for the full design). Receives ONE
// init message, builds its own Shahd instance from the same Config, re-points every parameter at
// the SHARED weight memory (and its private grad slab), then parks in a blocking Atomics loop with
// two phases per step:
//   COMPUTE — zero my slab, run my sequences (pretrain ForwardBackward or SFT SftForwardBackward,
//             fixed by Init.Mode), report the loss sum.
//   REDUCE  — fold my disjoint element range of EVERY worker's slab into the shared main-grad
//             buffer in fixed slab order, scaled by 1/BatchSize. Assignment covers the range, so
//             the main thread never zeroes or scales anything.
// After init there is no postMessage traffic at all — the barrier is pure shared-memory signaling,
// which is what lets the main thread's accumulation stay synchronous.
//
// The model code is UNTOUCHED here on purpose (plan §5): a worker trains a full private model that
// happens to alias shared weights, so every op, the tape, and the loss behave exactly as in the
// sequential path.

import type { WorkerInit } from "./WorkerPool.ts";
import { CmdSlot, StatusSlot, SeqCountSlot, CmdCompute, CmdShutdown, CmdReduce, StatusDone, StatusError, LossOutSlot, InvScaleSlot, AliasParams } from "./WorkerPool.ts";
import { CreateRngStreams } from "../Random/SeededRng.ts";
import { Shahd } from "../Nn/Shahd.ts";
import { NumView } from "../Tensor/Tensor.ts";
import { ForwardBackward } from "./TrainingStep.ts";
import { SftForwardBackward } from "../Sft/SftStep.ts";
import { ActivateFromConfig } from "../ComputeBackend/BackendSelector.ts";

// Worker-global surface, typed narrowly (no DOM lib in this project). The lowercase names are the
// Web Worker platform API — not ours to rename.
const WorkerScope = globalThis as unknown as {
  onmessage: ((Event: { data: WorkerInit }) => void) | null;
  // eslint-disable-next-line @typescript-eslint/naming-convention
  postMessage: (Value: unknown) => void;
};

function RunLoop(Init: WorkerInit): void {
  // Same backend selection as the main thread (each JS realm has its own selector; the Go DLL
  // itself loads once per process). Init weights are immediately overwritten by the shared alias.
  ActivateFromConfig(Init.Config);
  const Model = new Shahd(Init.Config, CreateRngStreams(Init.Config.Training.Seed).InitRng);
  AliasParams(Model.Parameters(), Init.Weights, Init.GradSlabs[Init.MyIndex], false);

  const Ctl = new Int32Array(Init.Ctl);
  // Slab views share the run's storage precision (ActivateFromConfig above set it from the SAME
  // Config the main thread used, so both sides view the shared buffers identically).
  const MyGrads = NumView(Init.GradSlabs[Init.MyIndex]);
  const AllSlabs = Init.GradSlabs.map((Slab) => NumView(Slab));
  const MainGrad = NumView(Init.MainGrad);
  const Ids = new Int32Array(Init.Ids);
  const Second = new Int32Array(Init.Second);
  const Lens = new Int32Array(Init.Lens);
  const Loss = new Float64Array(Init.Loss);

  const Compute = (): void => {
    MyGrads.fill(0); // fresh accumulation window (the model's Grad arrays alias this slab)
    const SeqCount = Atomics.load(Ctl, SeqCountSlot);
    let LossSum = 0;
    for (let S = 0; S < SeqCount; S++) {
      const Len = Lens[S];
      const Base = S * Init.MaxLen;
      const IdsArr = Array.from(Ids.subarray(Base, Base + Len));
      if (Init.Mode === "sft") {
        const Mask = new Array<boolean>(Len);
        for (let I = 0; I < Len; I++) Mask[I] = Second[Base + I] !== 0;
        LossSum += SftForwardBackward(Model, { Ids: IdsArr, LossMask: Mask });
      } else {
        LossSum += ForwardBackward(Model, IdsArr, Array.from(Second.subarray(Base, Base + Len)));
      }
    }
    Loss[LossOutSlot] = LossSum;
  };

  const Reduce = (): void => {
    const From = Init.ReduceFrom;
    const To = Init.ReduceTo;
    const Inv = Loss[InvScaleSlot];
    const First = AllSlabs[0];
    for (let I = From; I < To; I++) MainGrad[I] = First[I];
    for (let S = 1; S < AllSlabs.length; S++) {
      const G = AllSlabs[S];
      for (let I = From; I < To; I++) MainGrad[I] += G[I];
    }
    for (let I = From; I < To; I++) MainGrad[I] *= Inv;
  };

  WorkerScope.postMessage("ready"); // delivery does not need this thread's event loop afterwards

  while (true) {
    Atomics.wait(Ctl, CmdSlot, 0);
    const Cmd = Atomics.exchange(Ctl, CmdSlot, 0);
    if (Cmd === CmdShutdown) return;
    if (Cmd !== CmdCompute && Cmd !== CmdReduce) continue; // spurious wake with no pending command

    try {
      if (Cmd === CmdCompute) Compute();
      else Reduce();
      Atomics.store(Ctl, StatusSlot, StatusDone);
    } catch (Err) {
      // The error itself cannot cross the barrier — log it here, signal the class of failure.
      console.error("TrainWorker: step phase failed:", Err);
      Atomics.store(Ctl, StatusSlot, StatusError);
    }
    Atomics.notify(Ctl, StatusSlot);
  }
}

WorkerScope.onmessage = (Event) => {
  WorkerScope.onmessage = null; // exactly one init; the loop below never yields back
  RunLoop(Event.data);
};
