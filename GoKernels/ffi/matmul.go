// Go cgo c-shared kernel for the IN-PROCESS FFI path (ADR-0002). Built as a DLL and dlopen'd from
// TypeScript via bun:ffi — synchronous, zero IPC, so it can be a drop-in ComputeBackend. Requires
// a working C toolchain (gcc); build via PowerShell where MinGW gcc works cleanly (Git Bash
// shadows the toolchain and breaks cgo).
//
//   Build:  go build -buildmode=c-shared -o ../matmul.dll ./   (run from GoKernels/ffi)
//
// OPTIMIZED kernel (Phase 7): the spike showed a naive triple loop loses to Bun's JIT. Two levers
// close the gap: (1) transpose B once so the inner product streams BOTH operands contiguously
// (cache locality — Bun's JS matmul walks B column-wise and eats cache misses), and (2) fan the
// row range out across goroutines (real multi-core parallelism vs single-threaded JS). The k-loop
// still accumulates in ascending order with a plain mul-then-add, so the result stays BIT-IDENTICAL
// to TsBackend (ComputeSpike asserts parity == 0).

package main

import "C"
import (
	"runtime"
	"sync"
	"unsafe"
)

// MatMulF64 computes out[M,N] = a[M,K] @ b[K,N], row-major. out is fully overwritten.
//
//export MatMulF64
func MatMulF64(a *C.double, b *C.double, out *C.double, m C.int, k C.int, n C.int) {
	M := int(m)
	K := int(k)
	N := int(n)
	A := unsafe.Slice((*float64)(unsafe.Pointer(a)), M*K)
	B := unsafe.Slice((*float64)(unsafe.Pointer(b)), K*N)
	O := unsafe.Slice((*float64)(unsafe.Pointer(out)), M*N)

	// (1) Transpose B -> BT[N,K] so the dot loop walks A[i,:] and BT[j,:] contiguously.
	BT := make([]float64, N*K)
	for p := 0; p < K; p++ {
		pN := p * N
		for j := 0; j < N; j++ {
			BT[j*K+p] = B[pN+j]
		}
	}

	// (2) Split the M rows across goroutines — one worker per CPU, capped by row count.
	Workers := runtime.NumCPU()
	if Workers > M {
		Workers = M
	}
	if Workers < 1 {
		Workers = 1
	}
	RowsPer := (M + Workers - 1) / Workers

	var Wg sync.WaitGroup
	for W := 0; W < Workers; W++ {
		Start := W * RowsPer
		if Start >= M {
			break
		}
		End := Start + RowsPer
		if End > M {
			End = M
		}
		Wg.Add(1)
		go func(RowStart, RowEnd int) {
			defer Wg.Done()
			for i := RowStart; i < RowEnd; i++ {
				iK := i * K
				iN := i * N
				for j := 0; j < N; j++ {
					jK := j * K
					var Sum float64
					for p := 0; p < K; p++ {
						Sum += A[iK+p] * BT[jK+p] // ascending p, mul-then-add: bit-parity with TS
					}
					O[iN+j] = Sum
				}
			}
		}(Start, End)
	}
	Wg.Wait()
}

func main() {}
