// The compute-backend seam (ADR-0002). A backend does pure numeric work on flat Float64 buffers
// with explicit shapes and ZERO knowledge of Tensor/autograd — the only thing that can cheaply
// cross an FFI or process boundary. The autograd tape always stays in TypeScript; only hot-op
// numeric bodies may be routed to a backend.
//
// This SYNC interface fits the model's forward path. TsBackend implements it today. A Go FFI
// backend (in-process, sync) is the target once a working C toolchain exists; the Go SUBPROCESS
// backend (GoBackend) is async and is exposed separately (see the ComputeSpike finding).

export interface ComputeBackend {
  /** Out[M,N] = A[M,K] @ B[K,N], row-major. */
  MatMul(A: Float64Array, B: Float64Array, M: number, K: number, N: number): Float64Array;
}
