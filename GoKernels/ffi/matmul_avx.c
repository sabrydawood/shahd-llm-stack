// AVX2 + FMA micro-kernel for the row range of a matmul. Lives in its own .c file because cgo forbids
// DEFINITIONS in the preamble of a file that uses //export (matmul.go only declares this).
//
// Why C and not Go: Go has no SIMD intrinsics — a Go kernel is scalar unless hand-written in Plan9
// assembly, which is why the previous scalar Go kernel ran at ~16 GFLOP/s on a CPU capable of ~500.
// gcc with -mavx2 -mfma gives 4 doubles per instruction plus fused multiply-add, verified present on
// this machine (Broadwell: -mavx2/-mfma both [enabled] under -march=native).
//
// Three levers stack here:
//   1. B is pre-transposed by the caller (BT[N,K]) so A[i,:] and BT[j,:] both stream contiguously.
//   2. Register blocking over 8 rows of A: each BT[j,:] vector is loaded ONCE and reused by all 8
//      accumulators, cutting BT memory traffic ~8x. (This kernel is memory-bandwidth bound without it:
//      the naive form re-reads all of BT for every row of A.)
//   3. 4-wide FMA down the k loop.
//
// NUMERICS: this is deliberately NOT bit-identical to the scalar TS path. Vectorizing the k loop keeps
// 4 partial sums that are combined at the end, and FMA rounds once instead of twice. Both change the
// last bits. That trade was made explicitly to buy the SIMD speedup; gradcheck's 1e-3 finite-difference
// tolerance remains the correctness guard.

#include <immintrin.h>
#include <stddef.h>

// The ISA target is pinned HERE rather than via #cgo CFLAGS because cgo rejects -mfma as an unsafe
// flag (go.dev/s/invalidflag). Pinning it in the source also keeps the build a plain
// `go build -buildmode=c-shared` with no environment variables to remember.
#pragma GCC push_options
#pragma GCC target("avx2,fma")

// Horizontal sum of a 4-lane vector. Fixed lane order (0+2, 1+3, then the pair) so the result is
// deterministic run to run.
static inline double HSum256(__m256d V) {
    __m128d Lo = _mm256_castpd256_pd128(V);
    __m128d Hi = _mm256_extractf128_pd(V, 1);
    Lo = _mm_add_pd(Lo, Hi);
    __m128d Sh = _mm_unpackhi_pd(Lo, Lo);
    return _mm_cvtsd_f64(_mm_add_sd(Lo, Sh));
}

// O[i,j] = dot(A[i,:], BT[j,:]) for i in [RowStart, RowEnd), all j in [0, N).
void MatMulRowsAvx(const double *A, const double *BT, double *O,
                   int RowStart, int RowEnd, int K, int N) {
    const int Block = 8;
    int I = RowStart;

    for (; I + Block <= RowEnd; I += Block) {
        const double *A0 = A + (size_t)(I + 0) * K;
        const double *A1 = A + (size_t)(I + 1) * K;
        const double *A2 = A + (size_t)(I + 2) * K;
        const double *A3 = A + (size_t)(I + 3) * K;
        const double *A4 = A + (size_t)(I + 4) * K;
        const double *A5 = A + (size_t)(I + 5) * K;
        const double *A6 = A + (size_t)(I + 6) * K;
        const double *A7 = A + (size_t)(I + 7) * K;

        for (int J = 0; J < N; J++) {
            const double *Bt = BT + (size_t)J * K;
            __m256d S0 = _mm256_setzero_pd(), S1 = _mm256_setzero_pd();
            __m256d S2 = _mm256_setzero_pd(), S3 = _mm256_setzero_pd();
            __m256d S4 = _mm256_setzero_pd(), S5 = _mm256_setzero_pd();
            __m256d S6 = _mm256_setzero_pd(), S7 = _mm256_setzero_pd();

            int P = 0;
            for (; P + 4 <= K; P += 4) {
                __m256d Bv = _mm256_loadu_pd(Bt + P); // loaded once, reused by all 8 rows
                S0 = _mm256_fmadd_pd(_mm256_loadu_pd(A0 + P), Bv, S0);
                S1 = _mm256_fmadd_pd(_mm256_loadu_pd(A1 + P), Bv, S1);
                S2 = _mm256_fmadd_pd(_mm256_loadu_pd(A2 + P), Bv, S2);
                S3 = _mm256_fmadd_pd(_mm256_loadu_pd(A3 + P), Bv, S3);
                S4 = _mm256_fmadd_pd(_mm256_loadu_pd(A4 + P), Bv, S4);
                S5 = _mm256_fmadd_pd(_mm256_loadu_pd(A5 + P), Bv, S5);
                S6 = _mm256_fmadd_pd(_mm256_loadu_pd(A6 + P), Bv, S6);
                S7 = _mm256_fmadd_pd(_mm256_loadu_pd(A7 + P), Bv, S7);
            }

            double R0 = HSum256(S0), R1 = HSum256(S1), R2 = HSum256(S2), R3 = HSum256(S3);
            double R4 = HSum256(S4), R5 = HSum256(S5), R6 = HSum256(S6), R7 = HSum256(S7);
            for (; P < K; P++) { // K % 4 tail
                double Bv = Bt[P];
                R0 += A0[P] * Bv; R1 += A1[P] * Bv; R2 += A2[P] * Bv; R3 += A3[P] * Bv;
                R4 += A4[P] * Bv; R5 += A5[P] * Bv; R6 += A6[P] * Bv; R7 += A7[P] * Bv;
            }

            O[(size_t)(I + 0) * N + J] = R0;
            O[(size_t)(I + 1) * N + J] = R1;
            O[(size_t)(I + 2) * N + J] = R2;
            O[(size_t)(I + 3) * N + J] = R3;
            O[(size_t)(I + 4) * N + J] = R4;
            O[(size_t)(I + 5) * N + J] = R5;
            O[(size_t)(I + 6) * N + J] = R6;
            O[(size_t)(I + 7) * N + J] = R7;
        }
    }

    // Rows left over when the range is not a multiple of Block: same vectorized dot, one row at a time.
    for (; I < RowEnd; I++) {
        const double *Ai = A + (size_t)I * K;
        for (int J = 0; J < N; J++) {
            const double *Bt = BT + (size_t)J * K;
            __m256d S = _mm256_setzero_pd();
            int P = 0;
            for (; P + 4 <= K; P += 4) {
                S = _mm256_fmadd_pd(_mm256_loadu_pd(Ai + P), _mm256_loadu_pd(Bt + P), S);
            }
            double R = HSum256(S);
            for (; P < K; P++) R += Ai[P] * Bt[P];
            O[(size_t)I * N + J] = R;
        }
    }
}

#pragma GCC pop_options
