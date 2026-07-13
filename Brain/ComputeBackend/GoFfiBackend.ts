// In-process Go compute backend via bun:ffi (ADR-0002, the FAST owned path). Loads the cgo
// c-shared DLL and calls the matmul kernel synchronously with zero serialization — so unlike the
// subprocess GoBackend it IS a drop-in sync ComputeBackend. Requires GoKernels/matmul.dll (built
// from GoKernels/ffi via `go build -buildmode=c-shared`, done in PowerShell where MinGW gcc works).
//
// SPIKE FINDING: this proves the owned-Go FFI path works and is numerically exact, but a NAIVE Go
// matmul does NOT beat Bun's JIT-compiled TS matmul (~0.67x). The path only pays off with
// optimized kernels (cache tiling / SIMD / goroutine parallelism), not a straight port — so
// TsBackend stays the default until an optimized kernel exists.

import { dlopen, FFIType, ptr } from "bun:ffi";
import type { Pointer } from "bun:ffi";
import type { ComputeBackend } from "./ComputeBackend.ts";

type MatMulSymbol = (a: Pointer, b: Pointer, out: Pointer, m: number, k: number, n: number) => void;

export class GoFfiBackend implements ComputeBackend {
  private MatMulFn: MatMulSymbol;
  private CloseFn: () => void;

  constructor(LibPath = "GoKernels/matmul.dll") {
    const Lib = dlopen(LibPath, {
      MatMulF64: {
        args: [FFIType.ptr, FFIType.ptr, FFIType.ptr, FFIType.i32, FFIType.i32, FFIType.i32],
        returns: FFIType.void,
      },
    });
    this.MatMulFn = Lib.symbols.MatMulF64 as unknown as MatMulSymbol;
    this.CloseFn = (): void => {
      Lib.close();
    };
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

  Close(): void {
    this.CloseFn();
  }
}
