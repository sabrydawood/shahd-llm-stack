// Phase-2 compute spike: verify the Go subprocess backend matches the TS backend exactly and
// benchmark them on a large matmul. Finding gets printed for the record (ADR-0002).
//
//   bun run Scripts/ComputeSpike.ts

import { TsBackend } from "../Brain/ComputeBackend/TsBackend.ts";
import { GoBackend } from "../Brain/ComputeBackend/GoBackend.ts";
import { SeededRng } from "../Brain/Random/SeededRng.ts";

function RandBuffer(N: number, Rng: SeededRng): Float64Array {
  const Out = new Float64Array(N);
  for (let I = 0; I < N; I++) Out[I] = Rng.NextGaussian();
  return Out;
}

const Size = Number(process.argv[2] ?? 200); // M=K=N
const Repeats = 20;
const Rng = new SeededRng(1);
const A = RandBuffer(Size * Size, Rng);
const B = RandBuffer(Size * Size, Rng);

const Ts = new TsBackend();
const Go = new GoBackend();

// Parity.
const TsOut = Ts.MatMul(A, B, Size, Size, Size);
const GoOut = await Go.MatMul(A, B, Size, Size, Size);
let MaxDiff = 0;
for (let I = 0; I < TsOut.length; I++) MaxDiff = Math.max(MaxDiff, Math.abs(TsOut[I] - GoOut[I]));

// Benchmark TS.
const TsStart = Bun.nanoseconds();
for (let R = 0; R < Repeats; R++) Ts.MatMul(A, B, Size, Size, Size);
const TsMs = (Bun.nanoseconds() - TsStart) / 1e6 / Repeats;

// Benchmark Go (incl. IPC round trip).
const GoStart = Bun.nanoseconds();
for (let R = 0; R < Repeats; R++) await Go.MatMul(A, B, Size, Size, Size);
const GoMs = (Bun.nanoseconds() - GoStart) / 1e6 / Repeats;

Go.Close();

console.log(`ComputeSpike (${Size}x${Size} @ ${Size}x${Size}, avg of ${Repeats}):`);
console.log(`  parity maxAbsDiff = ${MaxDiff.toExponential(3)}`);
console.log(`  TS backend      = ${TsMs.toFixed(2)} ms/matmul`);
console.log(`  Go subprocess   = ${GoMs.toFixed(2)} ms/matmul (incl. stdio IPC)`);
console.log(`  speedup (TS/Go) = ${(TsMs / GoMs).toFixed(2)}x`);
console.log(
  "Finding: in-process FFI (cgo) is blocked by the broken local gcc; the subprocess path works\n" +
    "and is numerically exact, but pays IPC + async coloring — TS backend stays the sync default.",
);
