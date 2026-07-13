/* eslint-disable @typescript-eslint/no-explicit-any */
// Owned WebGPU (WGSL) Float32 matmul (M5). `any` is used deliberately: the WebGPU API has no type
// definitions in this runtime, and the code only executes where `navigator.gpu` exists.
//
// TWO HONEST CAVEATS:
//   1. WebGPU is ASYNC (adapter/device request, buffer mapAsync), while the ComputeBackend seam
//      (Ops/MatMul) is SYNC. So this is NOT a drop-in sync backend — wiring GPU into the forward
//      hot path requires making that path async (a separate, larger refactor). This is the owned
//      GPU kernel + probe + fallback that path will build on.
//   2. WebGPU/WGSL is Float32-only (no f64), matching the M2 mixed-precision path.
// Where WebGPU is absent (e.g. this runtime), callers use the CPU f32 fallback (AsyncCompute.ts).

const Nav = (globalThis as any).navigator;

export function WebGpuAvailable(): boolean {
  return typeof Nav === "object" && Nav !== null && Nav.gpu != null;
}

const Wgsl = `
struct Dims { m: u32, k: u32, n: u32 };
@group(0) @binding(0) var<storage, read> a: array<f32>;
@group(0) @binding(1) var<storage, read> b: array<f32>;
@group(0) @binding(2) var<storage, read_write> outp: array<f32>;
@group(0) @binding(3) var<uniform> dims: Dims;
@compute @workgroup_size(8, 8)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  let row = gid.x; let col = gid.y;
  if (row >= dims.m || col >= dims.n) { return; }
  var sum = 0.0;
  for (var p: u32 = 0u; p < dims.k; p = p + 1u) {
    sum = sum + a[row * dims.k + p] * b[p * dims.n + col];
  }
  outp[row * dims.n + col] = sum;
}`;

/** Compute Out[M,N] = A[M,K] @ B[K,N] in Float32 on the GPU. Throws if WebGPU is unavailable. */
export async function WebGpuMatMul(A: Float64Array, B: Float64Array, M: number, K: number, N: number): Promise<Float64Array> {
  const Gpu = Nav?.gpu;
  if (Gpu == null) throw new Error("WebGPU unavailable");
  const Adapter = await Gpu.requestAdapter();
  if (Adapter == null) throw new Error("no GPU adapter");
  const Device = await Adapter.requestDevice();
  const U = (globalThis as any).GPUBufferUsage;

  const Af = Float32Array.from(A);
  const Bf = Float32Array.from(B);
  const MakeStorage = (Data: Float32Array): any => {
    const Buf = Device.createBuffer({ size: Data.byteLength, usage: U.STORAGE | U.COPY_DST });
    Device.queue.writeBuffer(Buf, 0, Data);
    return Buf;
  };
  const ABuf = MakeStorage(Af);
  const BBuf = MakeStorage(Bf);
  const OutBuf = Device.createBuffer({ size: M * N * 4, usage: U.STORAGE | U.COPY_SRC });
  const DimBuf = Device.createBuffer({ size: 16, usage: U.UNIFORM | U.COPY_DST });
  Device.queue.writeBuffer(DimBuf, 0, new Uint32Array([M, K, N, 0]));
  const ReadBuf = Device.createBuffer({ size: M * N * 4, usage: U.MAP_READ | U.COPY_DST });

  const Module = Device.createShaderModule({ code: Wgsl });
  const Pipeline = Device.createComputePipeline({ layout: "auto", compute: { module: Module, entryPoint: "main" } });
  const Bind = Device.createBindGroup({
    layout: Pipeline.getBindGroupLayout(0),
    entries: [
      { binding: 0, resource: { buffer: ABuf } },
      { binding: 1, resource: { buffer: BBuf } },
      { binding: 2, resource: { buffer: OutBuf } },
      { binding: 3, resource: { buffer: DimBuf } },
    ],
  });

  const Encoder = Device.createCommandEncoder();
  const Pass = Encoder.beginComputePass();
  Pass.setPipeline(Pipeline);
  Pass.setBindGroup(0, Bind);
  Pass.dispatchWorkgroups(Math.ceil(M / 8), Math.ceil(N / 8));
  Pass.end();
  Encoder.copyBufferToBuffer(OutBuf, 0, ReadBuf, 0, M * N * 4);
  Device.queue.submit([Encoder.finish()]);

  await ReadBuf.mapAsync(1 /* GPUMapMode.READ */);
  const Out = Float64Array.from(new Float32Array(ReadBuf.getMappedRange().slice(0)));
  ReadBuf.unmap();
  Device.destroy?.();
  return Out;
}
