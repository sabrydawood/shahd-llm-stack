// Unbiased pass@k estimator (Chen et al., "Evaluating Large Language Models Trained on Code").
// Given N samples of which C pass the tests, the probability that at least one of a random k-subset
// passes is 1 - C(N-C, k)/C(N, k). Numerically stable product form.

export function PassAtK(N: number, C: number, K: number): number {
  if (K > N) throw new Error(`PassAtK: k (${K}) cannot exceed n (${N})`);
  if (N - C < K) return 1; // fewer than k failing samples => a k-subset must contain a pass
  let Product = 1;
  for (let I = 0; I < K; I++) Product *= (N - C - I) / (N - I);
  return 1 - Product;
}
