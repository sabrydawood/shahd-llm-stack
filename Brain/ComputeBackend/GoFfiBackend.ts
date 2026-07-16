// In-process Go compute backend via bun:ffi (ADR-0002, the FAST owned path). Loads the cgo
// c-shared DLL and calls the matmul kernel synchronously with zero serialization — so unlike the
// subprocess GoBackend it IS a drop-in sync ComputeBackend. Requires GoKernels/matmul.dll (built
// from GoKernels/ffi via `go build -buildmode=c-shared`, done in PowerShell where MinGW gcc works).
//
// FINDING (Phase 7, updated): a NAIVE Go port lost to Bun's JIT (~0.67x), but the OPTIMIZED kernel
// (B-transpose for cache locality + goroutine row-parallelism, see GoKernels/ffi/matmul.go) now
// WINS decisively and bit-exactly — measured ~2.0x at 128, ~6.7x at 256, ~7.8x at 512 (parity
// 0.0e+0). It stays an OPT-IN accelerator, not the default: it needs the prebuilt DLL (gcc, built
// via PowerShell), so TsBackend remains the zero-dependency fallback the hot path can always use.
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
