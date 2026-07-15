// STaR / RLVR round: sample K candidate solutions per coding problem from a trained model, KEEP only
// the ones whose code passes the sandboxed tests (execution = ground-truth verifiable reward), dedup
// them, and write the passing (prompt, solution) pairs as the next SFT round's data. This is the
// data-efficient reasoning lever: the model teaches itself from its OWN verified-correct outputs — no
// new human data, exactly "capability beyond dataset size".
//
// HONEST SCOPE: a toy-scale model will pass few/none of these problems; the yield is reported truthfully
// (0 passing is the expected result at current scale, NOT a bug). The loop + executor verifier are real
// and complete — point this at a capable model and it produces real SFT data for the next round.
//
//   bun run Scripts/StarRound.ts --Checkpoint=Checkpoints/Corpus.ckpt --K=8 --Out=Corpus/StarRound1.json

import { writeFileSync } from "node:fs";
import { LoadRunnableModel } from "../Brain/Checkpoint/LoadRunnableModel.ts";
import { Generate } from "../Brain/Sampling/Generate.ts";
import { DefaultSampling } from "../Brain/Sampling/Sampler.ts";
import { SeededRng } from "../Brain/Random/SeededRng.ts";
import { CodingProblems } from "../Brain/Eval/ProblemSet.ts";
import { CollectPassing } from "../Brain/Rl/RejectionSampling.ts";
import { ReadArg } from "./ScriptArgs.ts";

const CkptPath = ReadArg("--Checkpoint=", "Checkpoints/Corpus.ckpt");
const K = Math.max(1, Number(ReadArg("--K=", "8")));
const OutPath = ReadArg("--Out=", "Corpus/StarRound.json");
const Temperature = Number(ReadArg("--Temp=", "0.9"));
const MaxTokens = Number(ReadArg("--MaxTokens=", "160"));

const { Model, Tokenizer, Config } = LoadRunnableModel(CkptPath);
const Rng = new SeededRng(Config.Training.Seed + 12345);
console.log(`[star] model loaded from ${CkptPath} (vocab ${Tokenizer.VocabSize}); sampling K=${K} @ temp=${Temperature} over ${CodingProblems.length} problems`);

type Solved = { Problem: string; Prompt: string; Solutions: string[] };
const Collected: Solved[] = [];
let TotalPassing = 0;
let SolvedCount = 0;

for (const P of CodingProblems) {
  const PromptIds = Tokenizer.Encode(P.Prompt + "\n");
  const Candidates: string[] = [];
  for (let I = 0; I < K; I++) {
    const Out = Generate(Model, PromptIds, MaxTokens, { ...DefaultSampling, Temperature }, Rng);
    Candidates.push(Tokenizer.Decode(Out.slice(PromptIds.length)));
  }
  const Passing = CollectPassing(P, Candidates); // execute (candidate + tests); keep distinct passers
  TotalPassing += Passing.length;
  if (Passing.length > 0) {
    SolvedCount++;
    Collected.push({ Problem: P.Name, Prompt: P.Prompt, Solutions: Passing });
  }
  console.log(`[star] ${P.Name}: ${Passing.length}/${K} passing`);
}

console.log(`[star] solved ${SolvedCount}/${CodingProblems.length} problems; ${TotalPassing} verified solutions collected`);
if (TotalPassing === 0) {
  console.log("[star] NOTE: 0 passing — expected for a toy-scale model. The loop + executor verifier are real; a capable model yields SFT data here.");
}
writeFileSync(OutPath, JSON.stringify({ SolvedCount, TotalPassing, Round: Collected }, null, 2));
console.log(`[star] wrote ${OutPath}`);
