// Shahd Go compute worker (Phase 2, ADR-0002). A plain `go build` executable (NO cgo, so it
// needs no C toolchain — sidestepping the broken local gcc/cgo/FFI path) that talks to the
// TypeScript ComputeBackend over stdio with a tiny binary protocol. This is the "owned, no-Python"
// way to move heavy math off the pure-TS CPU loop.
//
//   Build:  go build -o GoKernels/worker.exe ./GoKernels
//
// Protocol (little-endian, matches JS TypedArray byte layout on x86):
//   request : int32 M, int32 K, int32 N, float64[M*K] A, float64[K*N] B
//   response: float64[M*N] Out   (Out = A @ B)
// Loops until stdin EOF.

package main

import (
	"bufio"
	"encoding/binary"
	"io"
	"math"
	"os"
)

func readFloats(r io.Reader, n int) ([]float64, error) {
	raw := make([]byte, n*8)
	if _, err := io.ReadFull(r, raw); err != nil {
		return nil, err
	}
	out := make([]float64, n)
	for i := 0; i < n; i++ {
		out[i] = math.Float64frombits(binary.LittleEndian.Uint64(raw[i*8:]))
	}
	return out, nil
}

func writeFloats(w io.Writer, buf []float64) error {
	raw := make([]byte, len(buf)*8)
	for i, v := range buf {
		binary.LittleEndian.PutUint64(raw[i*8:], math.Float64bits(v))
	}
	_, err := w.Write(raw)
	return err
}

func matMul(a, b []float64, m, k, n int) []float64 {
	out := make([]float64, m*n)
	for i := 0; i < m; i++ {
		iK := i * k
		iN := i * n
		for p := 0; p < k; p++ {
			av := a[iK+p]
			if av == 0 {
				continue
			}
			pN := p * n
			for j := 0; j < n; j++ {
				out[iN+j] += av * b[pN+j]
			}
		}
	}
	return out
}

func main() {
	r := bufio.NewReaderSize(os.Stdin, 1<<20)
	w := bufio.NewWriterSize(os.Stdout, 1<<20)
	var hdr [12]byte
	for {
		if _, err := io.ReadFull(r, hdr[:]); err != nil {
			return // EOF -> exit cleanly
		}
		m := int(int32(binary.LittleEndian.Uint32(hdr[0:4])))
		k := int(int32(binary.LittleEndian.Uint32(hdr[4:8])))
		n := int(int32(binary.LittleEndian.Uint32(hdr[8:12])))
		a, err := readFloats(r, m*k)
		if err != nil {
			return
		}
		b, err := readFloats(r, k*n)
		if err != nil {
			return
		}
		if err := writeFloats(w, matMul(a, b, m, k, n)); err != nil {
			return
		}
		if err := w.Flush(); err != nil {
			return
		}
	}
}
