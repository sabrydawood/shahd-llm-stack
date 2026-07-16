// The runtime-switchable compute backend selector (M2). `Ops/MatMul` asks `GetActiveBackend()` which
// backend to route through; **null means the inline f64 fast path** (the default — zero seam
// overhead, bit-identical, gradcheck-exact). `ActivateFromConfig` maps `Config.Compute` to a backend,
// probing availability (the Go FFI DLL) and falling back to CPU when `FallbackToCpu` is set, so a
// machine without a working GPU/FFI still runs. The active backend can be swapped at ANY time
// (e.g. turn a GPU off → CPU) via `SetActiveBackend`.

import type { ResolvedConfig } from "../Config/ConfigTypes.ts";
import type { ComputeBackend } from "./ComputeBackend.ts";
import { SetTensorPrecision } from "../Tensor/Tensor.ts";
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

/** Resolve Config.Compute to an active backend, with probe + CPU fallback. Returns what was chosen.
 *  Also sets the TENSOR STORAGE precision (SetTensorPrecision) — F32 is real storage now, not a
 *  per-call conversion backend: tensors allocate Float32Array and the inline op loops work on them
 *  unchanged (JS math is f64 and rounds on store), while GoFfi dispatches to the f32 C kernels. */
export function ActivateFromConfig(Config: ResolvedConfig): BackendChoice {
  const C = Config.Compute;
  SetTensorPrecision(C.Precision);

  if (C.Backend === "Ts") {
    SetActiveBackend(null); // the inline loops are precision-agnostic over the stored dtype
    return { Chosen: `Ts/${C.Precision} (inline)`, FellBack: false };
  }

  if (C.Backend === "GoFfi") {
    const Ffi = TryGoFfi();
    // An F32 run needs the f32 kernel trio; an older DLL without it must fall back rather than
    // feed Float32Array buffers to a kernel reading doubles.
    const Usable = Ffi !== null && (C.Precision === "F64" || Ffi.HasF32);
    if (Ffi !== null && Usable) {
      SetActiveBackend(Ffi); // GoFfiBackend reuses one cached DLL handle, so re-activating never reloads
      return { Chosen: `GoFfi/${C.Precision}`, FellBack: false };
    }
    if (!C.FallbackToCpu) throw new Error("GoFfi backend unavailable and Compute.FallbackToCpu is false");
    SetActiveBackend(null);
    return { Chosen: `CPU fallback (${C.Precision})`, FellBack: true };
  }

  // "Gpu" — reserved for M5; not built yet.
  if (!C.FallbackToCpu) throw new Error("GPU backend not built and Compute.FallbackToCpu is false");
  SetActiveBackend(null);
  return { Chosen: `CPU fallback (${C.Precision})`, FellBack: true };
}
