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
import type { NumArray, NumArrayCtor } from "../Tensor/Tensor.ts";

type MatMulSymbol = (a: Pointer, b: Pointer, out: Pointer, m: number, k: number, n: number) => void;
type MatMulAccSymbol = (a: Pointer, b: Pointer, out: Pointer, m: number, k: number, n: number, acc: number) => void;
type ThreadsSymbol = (n: number) => void;
type KernelSymbols = {
  MatMul: MatMulSymbol;
  Nt: MatMulAccSymbol | null;
  Tn: MatMulAccSymbol | null;
  // The f32 trio (8 SIMD lanes instead of 4). Null on DLLs built before the f32 kernels existed;
  // BackendSelector treats a null here as "GoFfi cannot serve an F32 run" and falls back to CPU.
  MatMul32: MatMulSymbol | null;
  Nt32: MatMulAccSymbol | null;
  Tn32: MatMulAccSymbol | null;
  Threads: ThreadsSymbol | null;
};

const MatMulDef = {
  args: [FFIType.ptr, FFIType.ptr, FFIType.ptr, FFIType.i32, FFIType.i32, FFIType.i32],
  returns: FFIType.void,
} as const;
const MatMulAccDef = {
  args: [FFIType.ptr, FFIType.ptr, FFIType.ptr, FFIType.i32, FFIType.i32, FFIType.i32, FFIType.i32],
  returns: FFIType.void,
} as const;
const ThreadsDef = { args: [FFIType.i32], returns: FFIType.void } as const;

// Strong, process-lifetime refs keyed by library path (see the header for why these are never closed).
const LoadedLibs = new Map<string, KernelSymbols>();

function LoadKernelSymbols(LibPath: string): KernelSymbols {
  const Cached = LoadedLibs.get(LibPath);
  if (Cached !== undefined) return Cached;

  let Symbols: KernelSymbols;
  try {
    const Lib = dlopen(LibPath, {
      MatMulF64: MatMulDef,
      MatMulNTF64: MatMulAccDef,
      MatMulTNF64: MatMulAccDef,
      MatMulF32: MatMulDef,
      MatMulNTF32: MatMulAccDef,
      MatMulTNF32: MatMulAccDef,
      SetKernelThreads: ThreadsDef,
    });
    Symbols = {
      MatMul: Lib.symbols.MatMulF64 as unknown as MatMulSymbol,
      Nt: Lib.symbols.MatMulNTF64 as unknown as MatMulAccSymbol,
      Tn: Lib.symbols.MatMulTNF64 as unknown as MatMulAccSymbol,
      MatMul32: Lib.symbols.MatMulF32 as unknown as MatMulSymbol,
      Nt32: Lib.symbols.MatMulNTF32 as unknown as MatMulAccSymbol,
      Tn32: Lib.symbols.MatMulTNF32 as unknown as MatMulAccSymbol,
      Threads: Lib.symbols.SetKernelThreads as unknown as ThreadsSymbol,
    };
  } catch {
    try {
      // DLL predating the f32 kernels — full f64 set only.
      const Lib = dlopen(LibPath, {
        MatMulF64: MatMulDef,
        MatMulNTF64: MatMulAccDef,
        MatMulTNF64: MatMulAccDef,
        SetKernelThreads: ThreadsDef,
      });
      Symbols = {
        MatMul: Lib.symbols.MatMulF64 as unknown as MatMulSymbol,
        Nt: Lib.symbols.MatMulNTF64 as unknown as MatMulAccSymbol,
        Tn: Lib.symbols.MatMulTNF64 as unknown as MatMulAccSymbol,
        MatMul32: null,
        Nt32: null,
        Tn32: null,
        Threads: Lib.symbols.SetKernelThreads as unknown as ThreadsSymbol,
      };
    } catch {
      // Oldest DLL without the backward entry points (or the throw is "no such file" — then this
      // dlopen throws too, which is the constructor's absent-DLL signal for TryGoFfi).
      const Lib = dlopen(LibPath, { MatMulF64: MatMulDef });
      Symbols = { MatMul: Lib.symbols.MatMulF64 as unknown as MatMulSymbol, Nt: null, Tn: null, MatMul32: null, Nt32: null, Tn32: null, Threads: null };
    }
  }
  LoadedLibs.set(LibPath, Symbols);
  return Symbols;
}

function BufPtr(Buf: NumArray): Pointer {
  const P = ptr(Buf);
  if (P === null) throw new Error("GoFfiBackend: null buffer pointer");
  return P;
}

export class GoFfiBackend implements ComputeBackend {
  private Symbols: KernelSymbols;

  /** True when the DLL exports the f32 kernel trio — BackendSelector requires this for F32 runs. */
  readonly HasF32: boolean;

  // Optional ComputeBackend capabilities — only assigned when the DLL exports the new symbols, so
  // `Backend.MatMulNtAcc === undefined` remains an honest feature probe for Ops/MatMul.
  MatMulNtAcc?: (A: NumArray, Bt: NumArray, Out: NumArray, M: number, K: number, N: number) => void;
  MatMulTnAcc?: (A: NumArray, DOut: NumArray, Db: NumArray, M: number, K: number, N: number) => void;

  /** PROCESS-GLOBAL cap on per-call goroutine fan-out (one Go runtime serves every JS thread).
   *  The training worker pool sets 1 around its dispatch — its JS workers ARE the parallelism —
   *  and 0 restores the all-cores default. Only defined when the DLL exports the symbol. */
  SetKernelThreads?: (N: number) => void;

  constructor(LibPath = "GoKernels/matmul.dll") {
    this.Symbols = LoadKernelSymbols(LibPath); // throws when the DLL is absent — TryGoFfi probes with this
    const { Nt, Tn, Nt32, Tn32, Threads } = this.Symbols;
    this.HasF32 = this.Symbols.MatMul32 !== null;
    if (Nt !== null) {
      // Dispatch by element type per call: every buffer of one call shares one width (the run's
      // precision), so probing the FIRST operand is sufficient and costs one instanceof.
      this.MatMulNtAcc = (A, Bt, Out, M, K, N) => {
        const Fn = A instanceof Float32Array ? Nt32 : Nt;
        if (Fn === null) throw new Error("GoFfiBackend: DLL lacks the f32 NT kernel — rebuild GoKernels");
        Fn(BufPtr(A), BufPtr(Bt), BufPtr(Out), M, K, N, 1);
      };
    }
    if (Tn !== null) {
      this.MatMulTnAcc = (A, DOut, Db, M, K, N) => {
        const Fn = A instanceof Float32Array ? Tn32 : Tn;
        if (Fn === null) throw new Error("GoFfiBackend: DLL lacks the f32 TN kernel — rebuild GoKernels");
        Fn(BufPtr(A), BufPtr(DOut), BufPtr(Db), M, K, N, 1);
      };
    }
    if (Threads !== null) {
      this.SetKernelThreads = (N) => Threads(N);
    }
  }

  MatMul(A: NumArray, B: NumArray, M: number, K: number, N: number): NumArray {
    const Out = new (A.constructor as NumArrayCtor)(M * N);
    this.MatMulInto(A, B, Out, M, K, N);
    return Out;
  }

  MatMulInto(A: NumArray, B: NumArray, Out: NumArray, M: number, K: number, N: number): void {
    const Fn = A instanceof Float32Array ? this.Symbols.MatMul32 : this.Symbols.MatMul;
    if (Fn === null) throw new Error("GoFfiBackend: DLL lacks the f32 forward kernel — rebuild GoKernels");
    Fn(BufPtr(A), BufPtr(B), BufPtr(Out), M, K, N); // kernel fully overwrites Out
  }
}
