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
// NUMERICS: no longer bit-identical to the TS path (see matmul_avx.c) — vectorizing the k loop keeps
// 4 partial sums and FMA rounds once instead of twice. This was an explicit trade for the SIMD win;
// gradcheck's 1e-3 tolerance is the correctness guard.

package main

/*
#cgo CFLAGS: -O3
void MatMulRowsAvx(const double *A, const double *BT, double *O, int RowStart, int RowEnd, int K, int N);
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

// Reusable B-transpose scratch. The kernel is called thousands of times per training step and a fresh
// make() per call was pure allocator/GC churn (hundreds of KB each, at this model's shapes).
var BtPool = sync.Pool{
	New: func() any {
		Buf := make([]float64, 0)
		return &Buf
	},
}

// MatMulF64 computes out[M,N] = a[M,K] @ b[K,N], row-major. out is fully overwritten.
//
//export MatMulF64
func MatMulF64(a *C.double, b *C.double, out *C.double, m C.int, k C.int, n C.int) {
	M := int(m)
	K := int(k)
	N := int(n)
	B := unsafe.Slice((*float64)(unsafe.Pointer(b)), K*N)

	// (1) Transpose B -> BT[N,K] once, so the dot loop streams A[i,:] and BT[j,:] contiguously and the
	// C kernel can vector-load both.
	BtRef := BtPool.Get().(*[]float64)
	if cap(*BtRef) < N*K {
		*BtRef = make([]float64, N*K)
	}
	BT := (*BtRef)[:N*K]
	for p := 0; p < K; p++ {
		pN := p * N
		for j := 0; j < N; j++ {
			BT[j*K+p] = B[pN+j]
		}
	}
	BtPtr := (*C.double)(unsafe.Pointer(&BT[0])) // Go memory, but C never retains it past the call

	// (2) Split the M rows across goroutines — one worker per CPU, capped by row count, each range
	// snapped UP to a RowBlock multiple so workers run the C kernel's blocked fast path instead of
	// straddling a block boundary (which would drop rows into its slower one-at-a-time remainder loop).
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
			C.MatMulRowsAvx(a, BtPtr, out, C.int(RowStart), C.int(RowEnd), C.int(K), C.int(N))
		}(Start, End)
	}
	Wg.Wait()
	BtPool.Put(BtRef) // safe: every worker has joined, so nothing still reads BT
}

func main() {}
