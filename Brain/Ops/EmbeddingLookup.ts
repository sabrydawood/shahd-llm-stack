// Embedding lookup: select rows of Table[V,C] by Ids -> [T,C]. Backward scatters gradient back
// to the looked-up rows with += so repeated ids across positions accumulate correctly.

import { Tensor } from "../Tensor/Tensor.ts";
import { Tape } from "../Tensor/Tape.ts";

export function EmbeddingLookup(Table: Tensor, Ids: number[]): Tensor {
  const T = Ids.length;
  const C = Table.Cols;
  const Out = new Tensor(T, C, undefined, [Table]);
  for (let I = 0; I < T; I++) {
    const Row = Ids[I];
    for (let J = 0; J < C; J++) Out.Data[I * C + J] = Table.Data[Row * C + J];
  }

  if (Tape.On) {
    Out.BackwardFn = () => {
      for (let I = 0; I < T; I++) {
        const Row = Ids[I];
        for (let J = 0; J < C; J++) Table.Grad[Row * C + J] += Out.Grad[I * C + J];
      }
    };
  }
  return Out;
}
