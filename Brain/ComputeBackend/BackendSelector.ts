// The runtime-switchable compute backend selector (M2). `Ops/MatMul` asks `GetActiveBackend()` which
// backend to route through; **null means the inline f64 fast path** (the default — zero seam
// overhead, bit-identical, gradcheck-exact). `ActivateFromConfig` maps `Config.Compute` to a backend,
// probing availability (the Go FFI DLL) and falling back to CPU when `FallbackToCpu` is set, so a
// machine without a working GPU/FFI still runs. The active backend can be swapped at ANY time
// (e.g. turn a GPU off → CPU) via `SetActiveBackend`.

import type { ResolvedConfig } from "../Config/ConfigTypes.ts";
import type { ComputeBackend } from "./ComputeBackend.ts";
import { TsBackendF32 } from "./TsBackendF32.ts";
import { GoFfiBackend } from "./GoFfiBackend.ts";

let Active: ComputeBackend | null = null; // null => Ops/MatMul uses its inline f64 fast path

/** The backend Ops routes through, or null for the inline f64 fast path. */
export function GetActiveBackend(): ComputeBackend | null {
  return Active;
}

/** Swap the active backend at runtime; pass null to return to the inline CPU f64 path. Switching only
 *  changes what NEW ops route through — a backend switched AWAY from must stay callable, because
 *  Ops/MatMul captures the active backend in the backward closure of every tape node built while it
 *  was active, and those closures run later (during Backward). This used to Close() the outgoing Go
 *  FFI handle, which unloaded the DLL out from under exactly those closures and segfaulted the whole
 *  process. GoFfiBackend now holds one cached, process-lifetime handle per path, so a switch has
 *  nothing to free and nothing to leak. */
export function SetActiveBackend(Backend: ComputeBackend | null): void {
  Active = Backend;
}

export type BackendChoice = { Chosen: string; FellBack: boolean };

function TryGoFfi(): GoFfiBackend | null {
  try {
    return new GoFfiBackend();
  } catch {
    return null; // DLL/toolchain unavailable
  }
}

function CpuFor(Precision: "F64" | "F32"): ComputeBackend | null {
  return Precision === "F32" ? new TsBackendF32() : null; // null = inline f64
}

/** Resolve Config.Compute to an active backend, with probe + CPU fallback. Returns what was chosen. */
export function ActivateFromConfig(Config: ResolvedConfig): BackendChoice {
  const C = Config.Compute;

  if (C.Backend === "Ts") {
    SetActiveBackend(CpuFor(C.Precision));
    return { Chosen: C.Precision === "F32" ? "Ts/F32" : "Ts/F64 (inline)", FellBack: false };
  }

  if (C.Backend === "GoFfi") {
    const Ffi = C.Precision === "F64" ? TryGoFfi() : null; // Go f32 kernel not built yet
    if (Ffi !== null) {
      SetActiveBackend(Ffi); // GoFfiBackend reuses one cached DLL handle, so re-activating never reloads
      return { Chosen: "GoFfi/F64", FellBack: false };
    }
    if (!C.FallbackToCpu) throw new Error("GoFfi backend unavailable and Compute.FallbackToCpu is false");
    SetActiveBackend(CpuFor(C.Precision));
    return { Chosen: `CPU fallback (${C.Precision})`, FellBack: true };
  }

  // "Gpu" — reserved for M5; not built yet.
  if (!C.FallbackToCpu) throw new Error("GPU backend not built and Compute.FallbackToCpu is false");
  SetActiveBackend(CpuFor(C.Precision));
  return { Chosen: `CPU fallback (${C.Precision})`, FellBack: true };
}
