// The compute-backend seam (ADR-0002). A backend does pure numeric work on flat Float64 buffers
// with explicit shapes and ZERO knowledge of Tensor/autograd — the only thing that can cheaply
// cross an FFI or process boundary. The autograd tape always stays in TypeScript; only hot-op
// numeric bodies may be routed to a backend.
//
// This SYNC interface fits the model's forward path. TsBackend implements it today. A Go FFI
// backend (in-process, sync) is the target once a working C toolchain exists; the Go SUBPROCESS
// backend (GoBackend) is async and is exposed separately (see the ComputeSpike finding).
//
// The OPTIONAL methods are plumbing for the backward pass: without them, Ops/MatMul must transpose
// B and A on the JS side (single-threaded) and add the returned buffers into the grads with scalar
// `+=` loops — measured overhead worth removing. A backend only declares them when it can honor the
// exact semantics with no hidden copies; Ops/MatMul falls back to the transpose path otherwise, so
// simple backends (TsBackend, TsBackendF32) stay three-line implementations.

export interface ComputeBackend {
  /** Out[M,N] = A[M,K] @ B[K,N], row-major. Allocates and returns the result. */
  MatMul(A: Float64Array, B: Float64Array, M: number, K: number, N: number): Float64Array;

  /** OPTIONAL zero-copy forward: writes A[M,K] @ B[K,N] into Out (fully overwritten), no allocation. */
  MatMulInto?(A: Float64Array, B: Float64Array, Out: Float64Array, M: number, K: number, N: number): void;

  /** OPTIONAL backward half dA: Out[M,N] += A[M,K] @ Btᵀ, where Bt is stored [N,K] row-major —
   *  i.e. the second operand arrives ALREADY transposed, so the backend must not transpose again. */
  MatMulNtAcc?(A: Float64Array, Bt: Float64Array, Out: Float64Array, M: number, K: number, N: number): void;

  /** OPTIONAL backward half dB: Db[K,N] += Aᵀ @ DOut, where A is [M,K] and DOut is [M,N] — the
   *  first operand is consumed transposed IN PLACE (no [K,M] copy of A may be materialized). */
  MatMulTnAcc?(A: Float64Array, DOut: Float64Array, Db: Float64Array, M: number, K: number, N: number): void;
}
