// Near-duplicate detection via MinHash (Phase 3). Raw code scrapes are full of near-dupes
// (vendored deps, forks, boilerplate); exact-hash dedup misses them, and duplicated data wastes
// training signal and raises verbatim-memorization risk (REVIEW.md). MinHash estimates Jaccard
// similarity of k-shingle sets cheaply.

const FnvOffset = 2166136261;
const FnvPrime = 16777619;

function HashString(Text: string, Seed: number): number {
  let H = (FnvOffset ^ Seed) >>> 0;
  for (let I = 0; I < Text.length; I++) {
    H ^= Text.charCodeAt(I);
    H = Math.imul(H, FnvPrime);
  }
  return H >>> 0;
}

/** Character k-gram shingles (whitespace-normalized) — the set MinHash approximates. */
export function Shingles(Text: string, K = 5): string[] {
  const Norm = Text.replace(/\s+/g, " ").trim();
  if (Norm.length <= K) return [Norm];
  const Out: string[] = [];
  for (let I = 0; I + K <= Norm.length; I++) Out.push(Norm.slice(I, I + K));
  return Out;
}

/** MinHash signature: for each of NumHashes seeds, the min hash over all shingles. */
export function MinHashSignature(Text: string, NumHashes = 64, K = 5): number[] {
  const Grams = Shingles(Text, K);
  const Sig = new Array<number>(NumHashes).fill(0xffffffff);
  for (const Gram of Grams) {
    for (let S = 0; S < NumHashes; S++) {
      const H = HashString(Gram, S);
      if (H < Sig[S]) Sig[S] = H;
    }
  }
  return Sig;
}

/** Estimated Jaccard similarity = fraction of signature positions that agree. */
export function EstimateJaccard(SigA: number[], SigB: number[]): number {
  const N = Math.min(SigA.length, SigB.length);
  let Equal = 0;
  for (let I = 0; I < N; I++) if (SigA[I] === SigB[I]) Equal++;
  return Equal / N;
}

/** Greedily group documents whose estimated Jaccard >= Threshold (O(n^2); fine for a shard). */
export function NearDuplicateGroups(Docs: string[], Threshold = 0.8, NumHashes = 64): number[][] {
  const Sigs = Docs.map((D) => MinHashSignature(D, NumHashes));
  const GroupOf = new Array<number>(Docs.length).fill(-1);
  const Groups: number[][] = [];
  for (let I = 0; I < Docs.length; I++) {
    if (GroupOf[I] !== -1) continue;
    const Group = [I];
    GroupOf[I] = Groups.length;
    for (let J = I + 1; J < Docs.length; J++) {
      if (GroupOf[J] === -1 && EstimateJaccard(Sigs[I], Sigs[J]) >= Threshold) {
        GroupOf[J] = Groups.length;
        Group.push(J);
      }
    }
    Groups.push(Group);
  }
  return Groups;
}

/** Keep one representative per near-duplicate group; return the indices to keep. */
export function DedupedIndices(Docs: string[], Threshold = 0.8): number[] {
  return NearDuplicateGroups(Docs, Threshold).map((G) => G[0]);
}
