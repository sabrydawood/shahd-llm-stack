// Explicit named barrel for the Ops layer (rule: no index.ts anywhere — a folder's public
// surface is a named file). Consumers import ops from here.

export { MatMul } from "./MatMul.ts";
export { Add } from "./Add.ts";
export { AddBias } from "./AddBias.ts";
export { Scale } from "./Scale.ts";
export { ReLU } from "./ReLU.ts";
export { Transpose } from "./Transpose.ts";
export { CausalMask } from "./CausalMask.ts";
export { SoftmaxRows } from "./SoftmaxRows.ts";
export { LayerNorm } from "./LayerNorm.ts";
export { EmbeddingLookup } from "./EmbeddingLookup.ts";
export { CrossEntropy } from "./CrossEntropy.ts";
export { ApplyRope } from "./RotaryEmbedding.ts";
export { RmsNorm } from "./RmsNorm.ts";
export { Gelu } from "./Gelu.ts";
export { Silu } from "./Silu.ts";
export { Mul } from "./Mul.ts";
