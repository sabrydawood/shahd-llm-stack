// Factory selecting the optimizer from config (single decision point).

import { Adam } from "./Adam.ts";
import { AdamW } from "./AdamW.ts";
import type { Tensor } from "../Tensor/Tensor.ts";
import type { ResolvedConfig } from "../Config/ConfigTypes.ts";

export function CreateOptimizer(Params: Tensor[], Config: ResolvedConfig): Adam {
  return Config.Optimizer.Kind === "AdamW" ? new AdamW(Params, Config) : new Adam(Params, Config);
}
