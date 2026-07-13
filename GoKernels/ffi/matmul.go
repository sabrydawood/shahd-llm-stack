// Go cgo c-shared kernel for the IN-PROCESS FFI path (ADR-0002). Built as a DLL and dlopen'd from
// TypeScript via bun:ffi — synchronous, zero IPC, so it can be a drop-in ComputeBackend. Requires
// a working C toolchain (gcc); build via PowerShell where MinGW gcc works cleanly (Git Bash
// shadows the toolchain and breaks cgo).
//
//   Build:  go build -buildmode=c-shared -o ../matmul.dll ./   (run from GoKernels/ffi)

package main

import "C"
import "unsafe"

// MatMulF64 computes out[M,N] = a[M,K] @ b[K,N], row-major. out must be zero-initialized.
//
//export MatMulF64
func MatMulF64(a *C.double, b *C.double, out *C.double, m C.int, k C.int, n C.int) {
	M := int(m)
	K := int(k)
	N := int(n)
	A := unsafe.Slice((*float64)(unsafe.Pointer(a)), M*K)
	B := unsafe.Slice((*float64)(unsafe.Pointer(b)), K*N)
	O := unsafe.Slice((*float64)(unsafe.Pointer(out)), M*N)
	for i := 0; i < M; i++ {
		iK := i * K
		iN := i * N
		for p := 0; p < K; p++ {
			av := A[iK+p]
			if av == 0 {
				continue
			}
			pN := p * N
			for j := 0; j < N; j++ {
				O[iN+j] += av * B[pN+j]
			}
		}
	}
}

func main() {}
