// An owned, deterministic, dependency-free text embedding (M3). Char-trigram FNV hashing into a
// fixed-dimension signed vector, L2-normalized — enough for cosine near-duplicate detection and
// "find similar" in the Foundry (and it stores cleanly in a pgvector column). Real semantic
// embeddings from the trained model can replace this behind the same shape later.

const FnvOffset = 2166136261;
const FnvPrime = 16777619;

function Fnv(Text: string): number {
  let H = FnvOffset >>> 0;
  for (let I = 0; I < Text.length; I++) {
    H ^= Text.charCodeAt(I);
    H = Math.imul(H, FnvPrime);
  }
  return H >>> 0;
}

/** L2-normalized char-trigram hashing embedding of dimension Dim. */
export function HashingEmbedding(Text: string, Dim = 256): number[] {
  const Vector = new Array<number>(Dim).fill(0);
  const Lower = Text.toLowerCase();
  for (let I = 0; I + 3 <= Lower.length; I++) {
    const Hash = Fnv(Lower.slice(I, I + 3));
    const Bucket = Hash % Dim;
    Vector[Bucket] += (Hash >>> 16) & 1 ? 1 : -1;
  }
  let Norm = 0;
  for (const V of Vector) Norm += V * V;
  Norm = Math.sqrt(Norm) || 1;
  return Vector.map((V) => V / Norm);
}

/** Cosine similarity of two equal-length vectors (dot product; assumes normalized inputs). */
export function CosineSimilarity(A: number[], B: number[]): number {
  const N = Math.min(A.length, B.length);
  let Dot = 0;
  for (let I = 0; I < N; I++) Dot += A[I] * B[I];
  return Dot;
}
