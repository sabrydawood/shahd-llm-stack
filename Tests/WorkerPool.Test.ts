import { test, expect, afterEach } from "bun:test";
import { LoadConfig } from "../Brain/Config/LoadConfig.ts";
import { CreateRngStreams, SeededRng } from "../Brain/Random/SeededRng.ts";
import { Shahd } from "../Brain/Nn/Shahd.ts";
import { InMemoryDataLoader } from "../Brain/Data/DataLoader.ts";
import { AccumulateGradients } from "../Brain/Training/GradAccumulation.ts";
import { CreateOptimizer } from "../Brain/Optim/OptimBarrel.ts";
import { CreateTrainWorkerPool } from "../Brain/Training/WorkerPool.ts";
import { SetActiveBackend, ActivateFromConfig } from "../Brain/ComputeBackend/BackendSelector.ts";
import { SetTensorPrecision } from "../Brain/Tensor/Tensor.ts";
import { SftForwardBackward } from "../Brain/Sft/SftStep.ts";
import type { TrainingSequence } from "../Brain/Sft/ChatTemplate.ts";
import type { ResolvedConfig } from "../Brain/Config/ConfigTypes.ts";

// The pool must be a DROP-IN for AccumulateGradients: same data order, same grad scaling, same
// mean loss — the only permitted difference is float addition ORDER (sequences are summed per
// worker, then across workers, instead of strictly sequentially). These tests pin exactly that:
// near-equality vs the sequential path, and BIT-equality between two pooled runs (determinism).
//
// Backend is forced to Ts/F64 (the inline path) so the tests run on machines without the Go DLL
// (CI) and prove the mechanism independent of any kernel. Worker threads select their own backend
// from the same Config, so they use the inline path too.

afterEach(() => {
  SetActiveBackend(null);
  SetTensorPrecision("F64"); // the F32 pool test flips the process-global storage precision
});

function TestConfig(): ResolvedConfig {
  return LoadConfig({
    Overrides: {
      Model: { EmbedDim: 32, NumLayers: 1, NumHeads: 2, BlockSize: 16, VocabSize: 64 },
      Training: { BatchSize: 4, Workers: 2, Seed: 7 },
      Compute: { Backend: "Ts", Precision: "F64" },
    },
    UseCli: false,
    UseEnv: false,
  });
}

function TestTokens(): number[] {
  const Rng = new SeededRng(99);
  const Tokens = new Array<number>(2000);
  for (let I = 0; I < Tokens.length; I++) Tokens[I] = Math.floor(Rng.NextFloat() * 64);
  return Tokens;
}

function ConcatGrads(Model: Shahd): Float64Array {
  const Params = Model.Parameters();
  let Total = 0;
  for (const P of Params) Total += P.Size;
  const Out = new Float64Array(Total);
  let Offset = 0;
  for (const P of Params) {
    Out.set(P.Grad, Offset);
    Offset += P.Size;
  }
  return Out;
}

test("the pool works under F32 storage: shared f32 slabs, parity vs the sequential F32 path", async () => {
  const Tokens = TestTokens();
  const F32Cfg = (): ResolvedConfig =>
    LoadConfig({
      Overrides: {
        Model: { EmbedDim: 32, NumLayers: 1, NumHeads: 2, BlockSize: 16, VocabSize: 64 },
        Training: { BatchSize: 4, Workers: 2, Seed: 7 },
        Compute: { Backend: "Ts", Precision: "F32" },
      },
      UseCli: false,
      UseEnv: false,
    });

  // Sequential F32 reference (ActivateFromConfig sets the storage precision BEFORE the model).
  const CfgA = F32Cfg();
  ActivateFromConfig(CfgA);
  const RngA = CreateRngStreams(CfgA.Training.Seed);
  const ModelA = new Shahd(CfgA, RngA.InitRng);
  const OptA = CreateOptimizer(ModelA.Parameters(), CfgA);
  const LoaderA = new InMemoryDataLoader(Tokens, CfgA.Model.BlockSize, RngA.DataRng);
  const LossA = AccumulateGradients(ModelA, OptA, LoaderA, CfgA.Training.BatchSize);

  // Pooled F32 run: the shared weight/grad slabs must be Float32 views on BOTH sides.
  const CfgB = F32Cfg();
  ActivateFromConfig(CfgB);
  const RngB = CreateRngStreams(CfgB.Training.Seed);
  const ModelB = new Shahd(CfgB, RngB.InitRng);
  const LoaderB = new InMemoryDataLoader(Tokens, CfgB.Model.BlockSize, RngB.DataRng);
  const Pool = await CreateTrainWorkerPool(ModelB, CfgB);
  try {
    expect(ModelB.Parameters()[0].Data).toBeInstanceOf(Float32Array); // really shared f32
    const LossB = Pool.Accumulate(LoaderB, CfgB.Training.BatchSize);
    // f32 accumulation + reorder tolerance (the F64 parity test pins 1e-9; f32 noise is ~1e-6/add).
    expect(Math.abs(LossB - LossA)).toBeLessThan(1e-3);
    const GradsA = ConcatGrads(ModelA);
    const GradsB = ConcatGrads(ModelB);
    let MaxDiff = 0;
    for (let I = 0; I < GradsA.length; I++) MaxDiff = Math.max(MaxDiff, Math.abs(GradsA[I] - GradsB[I]));
    expect(MaxDiff).toBeLessThan(1e-3);
  } finally {
    Pool.Dispose();
  }
});

test("pooled accumulation matches the sequential path within reorder tolerance", async () => {
  const Tokens = TestTokens();

  // Sequential reference.
  const CfgA = TestConfig();
  const RngA = CreateRngStreams(CfgA.Training.Seed);
  const ModelA = new Shahd(CfgA, RngA.InitRng);
  const OptA = CreateOptimizer(ModelA.Parameters(), CfgA);
  const LoaderA = new InMemoryDataLoader(Tokens, CfgA.Model.BlockSize, RngA.DataRng);
  const LossA = AccumulateGradients(ModelA, OptA, LoaderA, CfgA.Training.BatchSize);
  const GradsA = ConcatGrads(ModelA);

  // Pooled run: identical config/seed -> identical init weights and loader stream.
  const CfgB = TestConfig();
  const RngB = CreateRngStreams(CfgB.Training.Seed);
  const ModelB = new Shahd(CfgB, RngB.InitRng);
  const LoaderB = new InMemoryDataLoader(Tokens, CfgB.Model.BlockSize, RngB.DataRng);
  const Pool = await CreateTrainWorkerPool(ModelB, CfgB);
  try {
    expect(Pool.WorkerCount).toBe(2);
    const LossB = Pool.Accumulate(LoaderB, CfgB.Training.BatchSize);
    expect(Math.abs(LossB - LossA)).toBeLessThan(1e-9);

    const GradsB = ConcatGrads(ModelB);
    expect(GradsB.length).toBe(GradsA.length);
    let MaxRel = 0;
    for (let I = 0; I < GradsA.length; I++) {
      MaxRel = Math.max(MaxRel, Math.abs(GradsA[I] - GradsB[I]) / (Math.abs(GradsA[I]) + 1e-9));
    }
    expect(MaxRel).toBeLessThan(1e-9); // reordered f64 sums only — anything larger is a real bug
  } finally {
    Pool.Dispose();
  }
});

test("two pooled runs are bit-identical (fixed reduction order = determinism)", async () => {
  const Tokens = TestTokens();

  async function PooledGrads(): Promise<{ Loss: number; Grads: Float64Array }> {
    const Cfg = TestConfig();
    const Rng = CreateRngStreams(Cfg.Training.Seed);
    const Model = new Shahd(Cfg, Rng.InitRng);
    const Loader = new InMemoryDataLoader(Tokens, Cfg.Model.BlockSize, Rng.DataRng);
    const Pool = await CreateTrainWorkerPool(Model, Cfg);
    try {
      const Loss = Pool.Accumulate(Loader, Cfg.Training.BatchSize);
      return { Loss, Grads: ConcatGrads(Model) };
    } finally {
      Pool.Dispose();
    }
  }

  const First = await PooledGrads();
  const Second = await PooledGrads();
  expect(Second.Loss).toBe(First.Loss);
  let Mismatches = 0;
  for (let I = 0; I < First.Grads.length; I++) {
    if (First.Grads[I] !== Second.Grads[I]) Mismatches++;
  }
  expect(Mismatches).toBe(0);
});

test("the pretrain pool rejects sequences that do not match BlockSize", async () => {
  const Cfg = TestConfig();
  const Rng = CreateRngStreams(Cfg.Training.Seed);
  const Model = new Shahd(Cfg, Rng.InitRng);
  const Pool = await CreateTrainWorkerPool(Model, Cfg);
  try {
    const BadLoader = { GetSequence: () => ({ Ids: [1, 2, 3], Targets: [2, 3, 4] }) };
    expect(() => Pool.Accumulate(BadLoader, 1)).toThrow(/BlockSize/);
  } finally {
    Pool.Dispose();
  }
});

test("pooled SFT accumulation matches the sequential masked path within reorder tolerance", async () => {
  // Variable-length sequences with loss masks — the exact shape TrainSftChat feeds. The pool must
  // reproduce the sequential SftForwardBackward batch (same sequences, same 1/batch scaling) up to
  // float addition order, and be bit-deterministic across two runs.
  const MakeBatch = (): TrainingSequence[] => {
    const Rng = new SeededRng(55);
    const Lengths = [17, 9, 12, 5]; // BlockSize 16 -> SFT MaxLen 17; deliberately varied
    return Lengths.map((Len) => {
      const Ids = new Array<number>(Len);
      const LossMask = new Array<boolean>(Len);
      for (let I = 0; I < Len; I++) {
        Ids[I] = Math.floor(Rng.NextFloat() * 64);
        LossMask[I] = I >= Math.floor(Len / 2); // train on the second half (the "assistant" part)
      }
      return { Ids, LossMask };
    });
  };

  // Sequential reference — mirrors the TrainSftChat serial loop exactly.
  const CfgA = TestConfig();
  const RngA = CreateRngStreams(CfgA.Training.Seed);
  const ModelA = new Shahd(CfgA, RngA.InitRng);
  const OptA = CreateOptimizer(ModelA.Parameters(), CfgA);
  OptA.ZeroGrad();
  let LossSum = 0;
  const BatchA = MakeBatch();
  for (const Seq of BatchA) LossSum += SftForwardBackward(ModelA, Seq);
  const Inv = 1 / BatchA.length;
  for (const P of OptA.Params) for (let I = 0; I < P.Size; I++) P.Grad[I] *= Inv;
  const LossA = LossSum * Inv;
  const GradsA = ConcatGrads(ModelA);

  // Pooled run: identical init, same batch.
  const CfgB = TestConfig();
  const RngB = CreateRngStreams(CfgB.Training.Seed);
  const ModelB = new Shahd(CfgB, RngB.InitRng);
  const Pool = await CreateTrainWorkerPool(ModelB, CfgB, "sft");
  try {
    const LossB = Pool.AccumulateSft(MakeBatch());
    expect(Math.abs(LossB - LossA)).toBeLessThan(1e-9);
    const GradsB = ConcatGrads(ModelB);
    let MaxRel = 0;
    for (let I = 0; I < GradsA.length; I++) {
      MaxRel = Math.max(MaxRel, Math.abs(GradsA[I] - GradsB[I]) / (Math.abs(GradsA[I]) + 1e-9));
    }
    expect(MaxRel).toBeLessThan(1e-9);

    // Determinism: a second pooled step over the same batch on the same pool must be bit-identical.
    const LossC = Pool.AccumulateSft(MakeBatch());
    const GradsC = ConcatGrads(ModelB);
    expect(LossC).toBe(LossB);
    let Mismatches = 0;
    for (let I = 0; I < GradsB.length; I++) if (GradsB[I] !== GradsC[I]) Mismatches++;
    expect(Mismatches).toBe(0);
  } finally {
    Pool.Dispose();
  }
});
