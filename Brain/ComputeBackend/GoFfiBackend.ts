// In-process Go compute backend via bun:ffi (ADR-0002, the FAST owned path). Loads the cgo
// c-shared DLL and calls the matmul kernel synchronously with zero serialization — so unlike the
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
type MatMulLib = { symbols: { MatMulF64: unknown } };

// Strong, process-lifetime refs keyed by library path (see the header for why these are never closed).
const LoadedLibs = new Map<string, MatMulLib>();

function LoadMatMulSymbol(LibPath: string): MatMulSymbol {
  let Lib = LoadedLibs.get(LibPath);
  if (Lib === undefined) {
    Lib = dlopen(LibPath, {
      MatMulF64: {
        args: [FFIType.ptr, FFIType.ptr, FFIType.ptr, FFIType.i32, FFIType.i32, FFIType.i32],
        returns: FFIType.void,
      },
    }) as unknown as MatMulLib;
    LoadedLibs.set(LibPath, Lib);
  }
  return Lib.symbols.MatMulF64 as unknown as MatMulSymbol;
}

export class GoFfiBackend implements ComputeBackend {
  private MatMulFn: MatMulSymbol;

  constructor(LibPath = "GoKernels/matmul.dll") {
    this.MatMulFn = LoadMatMulSymbol(LibPath); // throws when the DLL is absent — TryGoFfi probes with this
  }

  MatMul(A: Float64Array, B: Float64Array, M: number, K: number, N: number): Float64Array {
    const Out = new Float64Array(M * N); // zero-initialized; the kernel accumulates
    const APtr = ptr(A);
    const BPtr = ptr(B);
    const OutPtr = ptr(Out);
    if (APtr === null || BPtr === null || OutPtr === null) throw new Error("GoFfiBackend: null buffer pointer");
    this.MatMulFn(APtr, BPtr, OutPtr, M, K, N);
    return Out;
  }
}
