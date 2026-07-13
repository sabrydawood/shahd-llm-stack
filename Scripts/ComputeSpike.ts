// Phase-2 compute spike: verify the Go subprocess backend matches the TS backend exactly and
// benchmark them on a large matmul. Finding gets printed for the record (ADR-0002).
//
//   bun run Scripts/ComputeSpike.ts

import { TsBackend } from "../Brain/ComputeBackend/TsBackend.ts";
import { GoBackend } from "../Brain/ComputeBackend/GoBackend.ts";
import { GoFfiBackend } from "../Brain/ComputeBackend/GoFfiBackend.ts";
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

// In-process FFI (if the cgo DLL was built).
let FfiMs = NaN;
let FfiDiff = NaN;
try {
  const Ffi = new GoFfiBackend();
  const FfiOut = Ffi.MatMul(A, B, Size, Size, Size);
  FfiDiff = 0;
  for (let I = 0; I < TsOut.length; I++) FfiDiff = Math.max(FfiDiff, Math.abs(TsOut[I] - FfiOut[I]));
  const FfiStart = Bun.nanoseconds();
  for (let R = 0; R < Repeats; R++) Ffi.MatMul(A, B, Size, Size, Size);
  FfiMs = (Bun.nanoseconds() - FfiStart) / 1e6 / Repeats;
  Ffi.Close();
} catch (Err) {
  console.log(`  (FFI backend unavailable: ${(Err as Error).message})`);
}

console.log(`ComputeSpike (${Size}x${Size} @ ${Size}x${Size}, avg of ${Repeats}):`);
console.log(`  TS backend       = ${TsMs.toFixed(2)} ms/matmul`);
console.log(`  Go subprocess    = ${GoMs.toFixed(2)} ms/matmul (parity ${MaxDiff.toExponential(1)}, incl. IPC)`);
if (!Number.isNaN(FfiMs)) {
  console.log(`  Go FFI in-proc   = ${FfiMs.toFixed(2)} ms/matmul (parity ${FfiDiff.toExponential(1)}, zero IPC)`);
  console.log(`  speedup FFI vs TS = ${(TsMs / FfiMs).toFixed(2)}x`);
}
