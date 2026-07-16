// In-process Go compute backend via bun:ffi (ADR-0002, the FAST owned path). Loads the cgo
// c-shared DLL and calls the matmul kernels synchronously with zero serialization — so unlike the
// subprocess GoBackend it IS a drop-in sync ComputeBackend. Requires GoKernels/matmul.dll (built
// from GoKernels/ffi via `go build -buildmode=c-shared`, done in PowerShell where MinGW gcc works).
//
// FINDING: a naive scalar Go port LOST to Bun's JIT, and even the tuned scalar Go kernel was ~1.4-1.9x
// slower per core than the JIT (measured at GOMAXPROCS=1) — Go's only edge was cheap goroutine fan-out,
// because Go has no SIMD intrinsics. The kernel's inner loop is therefore C (AVX2+FMA, see
// GoKernels/ffi/matmul_avx.c) with Go keeping the row fan-out: measured 3.0-4.9x over the old scalar
// kernel across this model's real shapes, ~1.95x on a whole training step. It stays an OPT-IN
// accelerator, not the default: it needs the prebuilt DLL (gcc, built via PowerShell), so TsBackend
// remains the zero-dependency fallback the hot path can always use.
//
// Besides the allocating MatMul, this backend implements the OPTIONAL ComputeBackend methods:
// MatMulInto (writes into the caller's buffer — kills one full-copy + one allocation per forward)
// and MatMulNtAcc/MatMulTnAcc (the two backward halves as native kernels — kills the JS-side
// single-threaded transposes and scalar `+=` grad loops). The NT/TN symbols only exist in DLLs
// built after they were added, so loading falls back to the MatMulF64-only symbol set and simply
// leaves the optional methods undefined — Ops/MatMul then uses its transpose path unchanged.
//
// NOT bit-identical to TsBackend any more (vectorizing the k loop keeps 4 partial sums and FMA rounds
// once, not twice; drift is ~1e-14 relative). That was a deliberate trade for the SIMD win. The guard
// is Tests/Compute.Test.ts, which checks this kernel against the TS reference within f64 accumulation
// tolerance on shapes that hit its edge paths — RunGradCheck never activates a backend, so it only
// ever exercises the inline TS path and does NOT cover this code.
//
// The DLL is opened ONCE per path and kept for the PROCESS LIFETIME — never unloaded. That is a
// CORRECTNESS requirement, not an optimization: Ops/MatMul captures the active backend inside every
// tape node's backward closure, so a node built while this backend was active still calls into the
// library long after a SetActiveBackend switch. Unloading left those closures jumping into freed code
// and killed the process with a SIGSEGV (unrecoverable — no catchable error, no stack). Caching the
// handle also makes repeated ActivateFromConfig calls leak nothing, which is what the old
// close-on-switch was there for.

import { dlopen, FFIType, ptr } from "bun:ffi";
import type { Pointer } from "bun:ffi";
import type { ComputeBackend } from "./ComputeBackend.ts";

type MatMulSymbol = (a: Pointer, b: Pointer, out: Pointer, m: number, k: number, n: number) => void;
type MatMulAccSymbol = (a: Pointer, b: Pointer, out: Pointer, m: number, k: number, n: number, acc: number) => void;
type KernelSymbols = { MatMul: MatMulSymbol; Nt: MatMulAccSymbol | null; Tn: MatMulAccSymbol | null };

const MatMulDef = {
  args: [FFIType.ptr, FFIType.ptr, FFIType.ptr, FFIType.i32, FFIType.i32, FFIType.i32],
  returns: FFIType.void,
} as const;
const MatMulAccDef = {
  args: [FFIType.ptr, FFIType.ptr, FFIType.ptr, FFIType.i32, FFIType.i32, FFIType.i32, FFIType.i32],
  returns: FFIType.void,
} as const;

// Strong, process-lifetime refs keyed by library path (see the header for why these are never closed).
const LoadedLibs = new Map<string, KernelSymbols>();

function LoadKernelSymbols(LibPath: string): KernelSymbols {
  const Cached = LoadedLibs.get(LibPath);
  if (Cached !== undefined) return Cached;

  let Symbols: KernelSymbols;
  try {
    const Lib = dlopen(LibPath, { MatMulF64: MatMulDef, MatMulNTF64: MatMulAccDef, MatMulTNF64: MatMulAccDef });
    Symbols = {
      MatMul: Lib.symbols.MatMulF64 as unknown as MatMulSymbol,
      Nt: Lib.symbols.MatMulNTF64 as unknown as MatMulAccSymbol,
      Tn: Lib.symbols.MatMulTNF64 as unknown as MatMulAccSymbol,
    };
  } catch {
    // Older DLL without the backward entry points (or the throw is "no such file" — then this
    // second dlopen throws too, which is the constructor's absent-DLL signal for TryGoFfi).
    const Lib = dlopen(LibPath, { MatMulF64: MatMulDef });
    Symbols = { MatMul: Lib.symbols.MatMulF64 as unknown as MatMulSymbol, Nt: null, Tn: null };
  }
  LoadedLibs.set(LibPath, Symbols);
  return Symbols;
}

function BufPtr(Buf: Float64Array): Pointer {
  const P = ptr(Buf);
  if (P === null) throw new Error("GoFfiBackend: null buffer pointer");
  return P;
}

export class GoFfiBackend implements ComputeBackend {
  private Symbols: KernelSymbols;

  // Optional ComputeBackend capabilities — only assigned when the DLL exports the new symbols, so
  // `Backend.MatMulNtAcc === undefined` remains an honest feature probe for Ops/MatMul.
  MatMulNtAcc?: (A: Float64Array, Bt: Float64Array, Out: Float64Array, M: number, K: number, N: number) => void;
  MatMulTnAcc?: (A: Float64Array, DOut: Float64Array, Db: Float64Array, M: number, K: number, N: number) => void;

  constructor(LibPath = "GoKernels/matmul.dll") {
    this.Symbols = LoadKernelSymbols(LibPath); // throws when the DLL is absent — TryGoFfi probes with this
    const { Nt, Tn } = this.Symbols;
    if (Nt !== null) {
      this.MatMulNtAcc = (A, Bt, Out, M, K, N) => Nt(BufPtr(A), BufPtr(Bt), BufPtr(Out), M, K, N, 1);
    }
    if (Tn !== null) {
      this.MatMulTnAcc = (A, DOut, Db, M, K, N) => Tn(BufPtr(A), BufPtr(DOut), BufPtr(Db), M, K, N, 1);
    }
  }

  MatMul(A: Float64Array, B: Float64Array, M: number, K: number, N: number): Float64Array {
    const Out = new Float64Array(M * N);
    this.MatMulInto(A, B, Out, M, K, N);
    return Out;
  }

  MatMulInto(A: Float64Array, B: Float64Array, Out: Float64Array, M: number, K: number, N: number): void {
    this.Symbols.MatMul(BufPtr(A), BufPtr(B), BufPtr(Out), M, K, N); // kernel fully overwrites Out
  }
}
