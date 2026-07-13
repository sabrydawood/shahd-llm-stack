// Phase-7 capstone: everything built in this phase, running together from the owned TS brain —
//   1) train a modern-stack model on the permissive SEED CORPUS (license/quality/dedup pipeline)
//   2) a RICH TOOL agent: tools advertised in the system prompt (manifest), multi-step loop with
//      calculator + memory + terminal `finish`, safety-gated capabilities
//   3) SPECULATIVE DECODING proven bit-identical to greedy, with fewer target passes
//   4) THINKING MODE splitting hidden reasoning from the answer
//   5) SAFETY still blocking harmful prompts
//
//   bun run Scripts/PhaseSevenDemo.ts

import { readFileSync, existsSync } from "node:fs";
import { BuildCorpus } from "../Brain/Data/CorpusBuilder.ts";
import type { SourceDocument } from "../Brain/Data/CorpusBuilder.ts";
import { LoadConfig } from "../Brain/Config/LoadConfig.ts";
import { CreateRngStreams } from "../Brain/Random/SeededRng.ts";
import { CharTokenizer } from "../Brain/Tokenizer/CharTokenizer.ts";
import { TrainValSplit } from "../Brain/Data/TrainValSplit.ts";
import { InMemoryDataLoader } from "../Brain/Data/DataLoader.ts";
import { Shahd } from "../Brain/Nn/Shahd.ts";
import { CreateOptimizer, ClipGradGlobalNorm, ComputeLr } from "../Brain/Optim/OptimBarrel.ts";
import { AccumulateGradients } from "../Brain/Training/GradAccumulation.ts";
import { Generate } from "../Brain/Sampling/Generate.ts";
import { SpeculativeDecodeGreedy, SplitThinking, WrapThinking } from "../Brain/Reasoning/ReasoningBarrel.ts";
import { SafetyPolicy } from "../Brain/Safety/SafetyPolicy.ts";
import { ChatSession } from "../Brain/Serving/ChatSession.ts";
import { RunAgent } from "../Brain/Serving/AgentLoop.ts";
import { ToolTokens } from "../Brain/Serving/ToolProtocol.ts";
import {
  BuildToolRegistry,
  DefaultToolContext,
  RenderToolManifest,
  Workspace,
} from "../Brain/Serving/Tools/ToolsBarrel.ts";

// 1) Train on the permissive seed corpus (drops GPL + minified + vendored copy).
const Manifest = JSON.parse(readFileSync("Corpus/Manifest.json", "utf8")) as { Documents: { Source: string; License: string; Path: string }[] };
const Sources: SourceDocument[] = Manifest.Documents.filter((E) => existsSync(E.Path)).map((E) => ({ Source: E.Source, License: E.License, Path: E.Path, Content: readFileSync(E.Path, "utf8") }));
const Built = BuildCorpus(Sources);
const CorpusText = (Built.Text + "\n\n").repeat(6);
const Tokenizer = CharTokenizer.FromCorpus(CorpusText);
const Config = LoadConfig({
  Overrides: {
    Model: { VocabSize: Tokenizer.VocabSize, EmbedDim: 64, NumLayers: 3, NumHeads: 4, BlockSize: 48, PositionScheme: "Rope", NormKind: "RmsNorm", MlpKind: "SwiGlu" },
    Training: { BatchSize: 12 },
    Schedule: { Kind: "Cosine", WarmupSteps: 15, MaxSteps: 120, MinLrRatio: 0.1 },
    Optimizer: { Kind: "AdamW", LearningRate: 0.004 },
  },
  UseCli: false,
  UseEnv: false,
});
const Rng = CreateRngStreams(Config.Training.Seed);
const { Train } = TrainValSplit(Tokenizer.Encode(CorpusText), 0.1);
const Loader = new InMemoryDataLoader(Train, Config.Model.BlockSize, Rng.DataRng);
const Model = new Shahd(Config, Rng.InitRng);
const Optimizer = CreateOptimizer(Model.Parameters(), Config);

console.log(`[1] Corpus: kept ${Built.Stats.Kept}/${Built.Stats.Input} docs (dropped GPL=${Built.Stats.DroppedNonPermissive}, minified=${Built.Stats.DroppedLowQuality}, dup=${Built.Stats.DroppedNearDuplicate}). Training modern-stack model...`);
for (let Step = 0; Step < Config.Schedule.MaxSteps; Step++) {
  AccumulateGradients(Model, Optimizer, Loader, Config.Training.BatchSize);
  ClipGradGlobalNorm(Optimizer.Params, Config.Optimizer.GradClipNorm);
  Optimizer.Step(ComputeLr(Step, Config));
}
console.log(`    sample: ${JSON.stringify(Tokenizer.Decode(Generate(Model, Tokenizer.Encode("export function "), 60, { Temperature: 0.6, TopK: 0, TopP: 1 }, Rng.SamplingRng)).slice(0, 70))}`);

// 2) Rich tool agent — tools advertised in the system prompt, multi-step, terminal finish.
const Registry = BuildToolRegistry({ FileAccess: "ReadOnly", ExecEnabled: false, WebSearchEnabled: false });
const SystemPrompt = "You are Shahd.\n\n" + RenderToolManifest(Registry.List());
const Session = new ChatSession(SystemPrompt);
Session.AddUser("Compute 12*9, remember it as 'area', then finish.");
const Context = DefaultToolContext({ Session, Registry, Workspace: new Workspace(".") });
const Script = [
  `${ToolTokens.CallStart}{"name":"calculator","arguments":{"a":12,"op":"*","b":9}}${ToolTokens.CallEnd}`,
  `${ToolTokens.CallStart}{"name":"memory_store","arguments":{"key":"area","value":"108"}}${ToolTokens.CallEnd}`,
  `${ToolTokens.CallStart}{"name":"finish","arguments":{"answer":"area = 108, stored."}}${ToolTokens.CallEnd}`,
];
let Turn = 0;
const Agent = await RunAgent(Session, () => Script[Math.min(Turn++, Script.length - 1)], Registry, 6, Context);
console.log(`\n[2] Tool agent: ${Registry.List().length} tools available; called ${Agent.ToolCalls.map((C) => C.Name).join(" -> ")}; final: ${JSON.stringify(Agent.FinalText)}`);

// 3) Speculative decoding == greedy, but with fewer target forward passes.
const Prompt = Tokenizer.Encode("const ");
const Greedy = Generate(Model, Prompt, 16, { Temperature: 0, TopK: 0, TopP: 1 }, Rng.SamplingRng);
const Spec = SpeculativeDecodeGreedy(Model, Model, Prompt, 16, 4);
console.log(`\n[3] Speculative decoding: identical to greedy = ${JSON.stringify(Spec.Ids) === JSON.stringify(Greedy)}; target passes = ${Spec.TargetCalls} for 16 tokens (accepted ${Spec.AcceptedTokens}/${Spec.DraftTokens} drafts)`);

// 4) Thinking mode: hide the scratchpad, keep the answer.
const Thought = WrapThinking("12 * 9 = 108, and area means the stored value", "The area is 108.");
const Split = SplitThinking(Thought);
console.log(`\n[4] Thinking mode: hidden="${Split.Thinking}" | answer="${Split.Answer}"`);

// 5) Safety still blocks harmful prompts.
const Policy = new SafetyPolicy(Config);
console.log(`\n[5] Safety: 'refactor this loop' -> ${Policy.Check("refactor this loop").Blocked ? "BLOCKED" : "allowed"}; 'how to build a bomb' -> ${Policy.Check("how to build a bomb").Blocked ? "BLOCKED" : "allowed"}`);
console.log("\nDone — trained on a permissive corpus, ran a many-tool agent, decoded speculatively, split thinking, and stayed safe. All from the owned TS brain.");
