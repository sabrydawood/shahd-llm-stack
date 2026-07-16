// Go cgo c-shared kernel for the IN-PROCESS FFI path (ADR-0002). Built as a DLL and dlopen'd from
// TypeScript via bun:ffi — synchronous, zero IPC, so it can be a drop-in ComputeBackend. Requires
// a working C toolchain (gcc); build via PowerShell where MinGW gcc works cleanly (Git Bash
// shadows the toolchain and breaks cgo).
//
//   Build:  go build -buildmode=c-shared -o ../matmul.dll ./   (run from GoKernels/ffi)
//
// This file is the THREADING + FFI shell only; the arithmetic lives in matmul_avx.c (AVX2+FMA). The
// split is forced by cgo: a file using //export may only DECLARE C functions in its preamble, never
// define them.
//
// Why the inner loop is C, not Go: Go has no SIMD intrinsics, so a Go kernel stays scalar (it would
// need Plan9 assembly). Measured, the old scalar Go kernel ran ~16 GFLOP/s on a CPU capable of ~500,
// and was actually SLOWER per core than Bun's JIT — its only edge was goroutine parallelism. C+AVX2
// closes that gap while Go keeps doing what it is good at here: cheap fan-out across cores.
//
// THREE ENTRY POINTS, one forward + the two halves of the matmul backward:
//   MatMulF64   out = a @ b                 (transposes b internally, overwrites out)
//   MatMulNTF64 out (+)= a @ btᵀ            (bt ALREADY [N,K] row-major — zero transpose; dA half)
//   MatMulTNF64 db  (+)= aᵀ @ dout          (dedicated column kernel — zero transpose; dB half)
// The NT/TN pair exists because the TS backward used to transpose B and A on the JS side (single-
// threaded) just so it could call the forward entry — pure overhead, since dOut @ Bᵀ wants exactly
// the [N,K]-style layout B is already stored in, and Aᵀ @ dOut can read A's columns in place.
//
// NUMERICS: no longer bit-identical to the TS path (see matmul_avx.c) — vectorizing the k loop keeps
// 4 partial sums and FMA rounds once instead of twice. This was an explicit trade for the SIMD win;
// gradcheck's 1e-3 tolerance is the correctness guard.

package main

/*
#cgo CFLAGS: -O3
void MatMulRowsAvx(const double *A, const double *BT, double *O, int RowStart, int RowEnd, int K, int N, int Acc);
void MatMulTnAvx(const double *A, const double *DOut, double *DB, int PStart, int PEnd, int M, int K, int N, int Acc);
*/
import "C"
import (
	"runtime"
	"sync"
	"unsafe"
)

// Rows of A the C kernel accumulates simultaneously against one BT row. Must match Block in
// matmul_avx.c — it is what makes the range split worth aligning.
const RowBlock = 8

// Transpose elements per goroutine (~32 KB of float64, about an L1 working set). Sets how wide the
// B-transpose fans out; see the worker-count comment in MatMulF64 for why it is size-driven.
const TransposeChunk = 4096

// FLOPs per goroutine for the TN fan-out (~50µs of FMA work per core). Same principle as
// TransposeChunk: worker count scales with WORK, not cores — measured on the transpose, fanning
// narrow shapes across all cores loses ~30% to spawn/join, so small backward shapes stay narrow.
const TnFlopChunk = 262144

// Reusable B-transpose scratch. The kernel is called thousands of times per training step and a fresh
// make() per call was pure allocator/GC churn (hundreds of KB each, at this model's shapes).
var BtPool = sync.Pool{
	New: func() any {
		Buf := make([]float64, 0)
		return &Buf
	},
}

// FanOutRows splits [0,M) into RowBlock-aligned ranges across goroutines and runs the C row kernel
// on each — one worker per CPU, capped by row count, each range snapped UP to a RowBlock multiple so
// workers run the C kernel's blocked fast path instead of straddling a block boundary (which would
// drop rows into its slower one-at-a-time remainder loop). bt must be [N,K] row-major.
func FanOutRows(a, bt, out *C.double, M, K, N, Acc int) {
	Workers := runtime.NumCPU()
	if Workers > M {
		Workers = M
	}
	if Workers < 1 {
		Workers = 1
	}
	RowsPer := (M + Workers - 1) / Workers
	RowsPer = ((RowsPer + RowBlock - 1) / RowBlock) * RowBlock

	var Wg sync.WaitGroup
	for Start := 0; Start < M; Start += RowsPer {
		End := Start + RowsPer
		if End > M {
			End = M
		}
		Wg.Add(1)
		go func(RowStart, RowEnd int) {
			defer Wg.Done()
			C.MatMulRowsAvx(a, bt, out, C.int(RowStart), C.int(RowEnd), C.int(K), C.int(N), C.int(Acc))
		}(Start, End)
	}
	Wg.Wait()
}

// TransposeBt fills BT[N,K] from B[K,N] IN PARALLEL: a transpose is pure memory traffic (65k+
// elements at this model's shapes) and running it on one thread before the workers start made it
// Amdahl's serial fraction INSIDE the kernel. Splitting by output row (j) gives each worker a
// contiguous BT write range, so cores do not fight over the same cache lines.
//
// Worker count scales with transpose SIZE, not CPU count. Measured: fanning the wide MLP/LM-head
// transposes (~65k elements) across all cores wins ~1.4-1.6x, but doing the same to the narrow
// attention projections (~4k elements) LOSES ~30% — goroutine spawn/join costs more than the
// transpose itself at that size. One worker per TransposeChunk elements lets each shape pick.
func TransposeBt(BT, B []float64, K, N int) {
	Workers := runtime.NumCPU()
	TWorkers := (N * K) / TransposeChunk
	if TWorkers > Workers {
		TWorkers = Workers
	}
	if TWorkers > N {
		TWorkers = N
	}
	if TWorkers < 1 {
		TWorkers = 1
	}
	JPer := (N + TWorkers - 1) / TWorkers
	var TWg sync.WaitGroup
	for J0 := 0; J0 < N; J0 += JPer {
		J1 := J0 + JPer
		if J1 > N {
			J1 = N
		}
		TWg.Add(1)
		go func(JStart, JEnd int) {
			defer TWg.Done()
			for j := JStart; j < JEnd; j++ {
				jK := j * K
				for p := 0; p < K; p++ {
					BT[jK+p] = B[p*N+j]
				}
			}
		}(J0, J1)
	}
	TWg.Wait()
}

// MatMulF64 computes out[M,N] = a[M,K] @ b[K,N], row-major. out is fully overwritten.
//
//export MatMulF64
func MatMulF64(a *C.double, b *C.double, out *C.double, m C.int, k C.int, n C.int) {
	M := int(m)
	K := int(k)
	N := int(n)
	B := unsafe.Slice((*float64)(unsafe.Pointer(b)), K*N)

	BtRef := BtPool.Get().(*[]float64)
	if cap(*BtRef) < N*K {
		*BtRef = make([]float64, N*K)
	}
	BT := (*BtRef)[:N*K]
	TransposeBt(BT, B, K, N)
	BtPtr := (*C.double)(unsafe.Pointer(&BT[0])) // Go memory, but C never retains it past the call

	FanOutRows(a, BtPtr, out, M, K, N, 0)
	BtPool.Put(BtRef) // safe: every worker has joined, so nothing still reads BT
}

// MatMulNTF64 computes out[M,N] (+)= a[M,K] @ btᵀ where bt is ALREADY stored [N,K] row-major —
// exactly the layout the row kernel streams, so NO transpose happens at all. This is the dA half
// of the matmul backward: dA[M,K] += dOut[M,N] @ Bᵀ, where the stored B[K,N] IS the [K,N]-shaped
// pre-transposed operand. acc != 0 accumulates into out instead of overwriting.
//
//export MatMulNTF64
func MatMulNTF64(a *C.double, bt *C.double, out *C.double, m C.int, k C.int, n C.int, acc C.int) {
	FanOutRows(a, bt, out, int(m), int(k), int(n), int(acc))
}

// MatMulTNF64 computes db[K,N] (+)= aᵀ @ dout where a is [M,K] and dout is [M,N] — the dB half of
// the matmul backward (dB += Aᵀ @ dOut), again with zero transposes: the C kernel reads a's column
// p in place while DB[p,:] stays hot in L1. Fan-out is over db ROWS (p ranges), so every goroutine
// writes a disjoint db range and only shares reads.
//
//export MatMulTNF64
func MatMulTNF64(a *C.double, dout *C.double, db *C.double, m C.int, k C.int, n C.int, acc C.int) {
	M := int(m)
	K := int(k)
	N := int(n)

	Workers := (M * K * N) / TnFlopChunk
	Cpus := runtime.NumCPU()
	if Workers > Cpus {
		Workers = Cpus
	}
	if Workers > K {
		Workers = K
	}
	if Workers < 1 {
		Workers = 1
	}
	PPer := (K + Workers - 1) / Workers

	var Wg sync.WaitGroup
	for P0 := 0; P0 < K; P0 += PPer {
		P1 := P0 + PPer
		if P1 > K {
			P1 = K
		}
		Wg.Add(1)
		go func(PStart, PEnd int) {
			defer Wg.Done()
			C.MatMulTnAvx(a, dout, db, C.int(PStart), C.int(PEnd), C.int(M), C.int(K), C.int(N), C.int(acc))
		}(P0, P1)
	}
	Wg.Wait()
}

func main() {}
