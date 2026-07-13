// Resolves the final config: DefaultConfig (Constants) -> JSON preset -> programmatic overrides
// -> CLI (--Section.Key=Value) -> Zod validation -> derivation -> deep-freeze + ConfigHash.
// The returned ResolvedConfig is immutable and self-describing (its full contents + hash are
// what gets embedded verbatim in every checkpoint).

import { readFileSync, existsSync } from "node:fs";
import { createHash } from "node:crypto";
import { DefaultConfig } from "./Constants.ts";
import { ValidateConfig } from "./ValidateConfig.ts";
import { DeriveConfig } from "./DeriveConfig.ts";
import type { ResolvedConfig, ConfigOverride, ShahdConfig } from "./ConfigTypes.ts";

type PlainObject = Record<string, unknown>;

export type LoadConfigOptions = {
  ConfigPath?: string; // JSON preset path (falls back to the SHAHD_CONFIG env var)
  Overrides?: ConfigOverride; // programmatic deep-partial override (used by tests/scripts)
  Argv?: string[]; // defaults to process.argv.slice(2)
  UseCli?: boolean; // default true
  UseEnv?: boolean; // default true (reads SHAHD_CONFIG for the preset path)
};

function IsPlainObject(Value: unknown): Value is PlainObject {
  return typeof Value === "object" && Value !== null && !Array.isArray(Value);
}

function DeepMerge(Base: PlainObject, Override: PlainObject): PlainObject {
  const Result: PlainObject = { ...Base };
  for (const Key of Object.keys(Override)) {
    const OverrideValue = Override[Key];
    const BaseValue = Result[Key];
    if (IsPlainObject(BaseValue) && IsPlainObject(OverrideValue)) {
      Result[Key] = DeepMerge(BaseValue, OverrideValue);
    } else if (OverrideValue !== undefined) {
      Result[Key] = OverrideValue;
    }
  }
  return Result;
}

function SetDeep(Target: PlainObject, Path: string[], Value: unknown): void {
  let Cursor = Target;
  for (let I = 0; I < Path.length - 1; I++) {
    const Key = Path[I];
    if (Key === undefined) return;
    if (!IsPlainObject(Cursor[Key])) Cursor[Key] = {};
    Cursor = Cursor[Key] as PlainObject;
  }
  const Last = Path[Path.length - 1];
  if (Last !== undefined) Cursor[Last] = Value;
}

function ParseCliOverrides(Argv: string[]): PlainObject {
  const Result: PlainObject = {};
  for (const Arg of Argv) {
    if (!Arg.startsWith("--")) continue;
    const Eq = Arg.indexOf("=");
    if (Eq === -1) continue;
    const Path = Arg.slice(2, Eq).split(".");
    const RawValue = Arg.slice(Eq + 1);
    let Parsed: unknown;
    try {
      Parsed = JSON.parse(RawValue); // coerces numbers/bools/JSON; falls back to the raw string
    } catch {
      Parsed = RawValue;
    }
    SetDeep(Result, Path, Parsed);
  }
  return Result;
}

function CanonicalStringify(Value: unknown): string {
  if (Array.isArray(Value)) {
    return "[" + Value.map(CanonicalStringify).join(",") + "]";
  }
  if (IsPlainObject(Value)) {
    const Keys = Object.keys(Value).sort();
    return "{" + Keys.map((K) => JSON.stringify(K) + ":" + CanonicalStringify(Value[K])).join(",") + "}";
  }
  return JSON.stringify(Value);
}

function ComputeConfigHash(Config: ShahdConfig): string {
  return createHash("sha256").update(CanonicalStringify(Config)).digest("hex").slice(0, 16);
}

function DeepFreeze<T>(Value: T): T {
  if (IsPlainObject(Value) || Array.isArray(Value)) {
    for (const Key of Object.keys(Value as PlainObject)) {
      DeepFreeze((Value as PlainObject)[Key]);
    }
    Object.freeze(Value);
  }
  return Value;
}

export function LoadConfig(Options: LoadConfigOptions = {}): ResolvedConfig {
  let Merged = DeepMerge({}, DefaultConfig as unknown as PlainObject); // clone the defaults

  const PresetPath =
    Options.ConfigPath ?? (Options.UseEnv === false ? undefined : process.env["SHAHD_CONFIG"]);
  if (PresetPath !== undefined && PresetPath !== "" && existsSync(PresetPath)) {
    const FileJson = JSON.parse(readFileSync(PresetPath, "utf8")) as PlainObject;
    Merged = DeepMerge(Merged, FileJson);
  }

  if (Options.Overrides !== undefined) {
    Merged = DeepMerge(Merged, Options.Overrides as PlainObject);
  }

  if (Options.UseCli !== false) {
    Merged = DeepMerge(Merged, ParseCliOverrides(Options.Argv ?? process.argv.slice(2)));
  }

  const Validated = ValidateConfig(Merged);
  const Resolved: ResolvedConfig = {
    ...Validated,
    Derived: DeriveConfig(Validated),
    ConfigHash: ComputeConfigHash(Validated),
  };
  return DeepFreeze(Resolved);
}
