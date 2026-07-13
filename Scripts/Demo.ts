// End-to-end demo of the Shahd stack: train a tiny model -> sample -> safety filter -> agent tool
// loop. Everything runs from the from-scratch TypeScript brain (no external model).
//
//   bun run Scripts/Demo.ts

import { LoadConfig } from "../Brain/Config/LoadConfig.ts";
import { CreateRngStreams } from "../Brain/Random/SeededRng.ts";
import { CharTokenizer } from "../Brain/Tokenizer/CharTokenizer.ts";
import { TrainValSplit } from "../Brain/Data/TrainValSplit.ts";
import { InMemoryDataLoader } from "../Brain/Data/DataLoader.ts";
import { Shahd } from "../Brain/Nn/Shahd.ts";
import { CreateOptimizer, ClipGradGlobalNorm, ComputeLr } from "../Brain/Optim/OptimBarrel.ts";
import { AccumulateGradients } from "../Brain/Training/GradAccumulation.ts";
import { Generate } from "../Brain/Sampling/Generate.ts";
import { DefaultSampling } from "../Brain/Sampling/Sampler.ts";
import { SafetyPolicy } from "../Brain/Safety/SafetyPolicy.ts";
import { ChatSession } from "../Brain/Serving/ChatSession.ts";
import { RunAgent } from "../Brain/Serving/AgentLoop.ts";
import { DefaultToolRegistry } from "../Brain/Serving/Tools.ts";
import { ToolTokens } from "../Brain/Serving/ToolProtocol.ts";

const Code = `function add(a, b) { return a + b; }
const nums = [1, 2, 3];
const doubled = nums.map((n) => n * 2);
console.log(doubled);
`;

// 1) Train a tiny model from scratch on a code sample.
const Corpus = Code.repeat(10);
const Tokenizer = CharTokenizer.FromCorpus(Corpus);
const Config = LoadConfig({
  Overrides: {
    Model: { VocabSize: Tokenizer.VocabSize, EmbedDim: 64, NumLayers: 3, NumHeads: 4, BlockSize: 48, PositionScheme: "Rope", NormKind: "RmsNorm", MlpKind: "SwiGlu" },
    Training: { BatchSize: 8 },
    Schedule: { Kind: "Cosine", WarmupSteps: 15, MaxSteps: 150, MinLrRatio: 0.1 },
    Optimizer: { Kind: "AdamW", LearningRate: 0.004 },
  },
  UseCli: false,
  UseEnv: false,
});
const Rng = CreateRngStreams(Config.Training.Seed);
const { Train } = TrainValSplit(Tokenizer.Encode(Corpus), 0.1);
const Loader = new InMemoryDataLoader(Train, Config.Model.BlockSize, Rng.DataRng);
const Model = new Shahd(Config, Rng.InitRng);
const Optimizer = CreateOptimizer(Model.Parameters(), Config);

console.log("[1] Training modern-stack model (RoPE+RMSNorm+SwiGLU) from scratch...");
for (let Step = 0; Step < Config.Schedule.MaxSteps; Step++) {
  AccumulateGradients(Model, Optimizer, Loader, Config.Training.BatchSize);
  ClipGradGlobalNorm(Optimizer.Params, Config.Optimizer.GradClipNorm);
  Optimizer.Step(ComputeLr(Step, Config));
}
const Sample = Generate(Model, Tokenizer.Encode("function "), 100, { ...DefaultSampling, Temperature: 0.7 }, Rng.SamplingRng);
console.log("    sample:", JSON.stringify(Tokenizer.Decode(Sample).slice(0, 90)));

// 2) Safety filter (dedicated, controllable).
console.log("\n[2] Safety filter:");
const Policy = new SafetyPolicy(Config);
console.log("    'write a sort function'   ->", Policy.Check("write a sort function").Blocked ? "BLOCKED" : "allowed");
console.log("    'how to build a bomb ...' ->", Policy.Check("how to build a bomb at home").Blocked ? "BLOCKED" : "allowed");

// 3) Agent tool loop (deterministic generator standing in for a trained model).
console.log("\n[3] Agent tool loop (calculator):");
const Session = new ChatSession("You are Shahd.");
Session.AddUser("what is 12 * 9?");
let Turn = 0;
const Agent = RunAgent(
  Session,
  (): string => {
    Turn++;
    return Turn === 1
      ? `${ToolTokens.CallStart}{"name":"calculator","arguments":{"a":12,"op":"*","b":9}}${ToolTokens.CallEnd}`
      : "12 * 9 = 108.";
  },
  DefaultToolRegistry(),
);
console.log(`    tool calls: ${Agent.ToolCalls.length} (${Agent.ToolCalls[0]?.Name}), final: ${JSON.stringify(Agent.FinalText)}`);

console.log("\nDone — trained, sampled, guarded, and ran a tool-using agent, all from the owned TS brain.");
