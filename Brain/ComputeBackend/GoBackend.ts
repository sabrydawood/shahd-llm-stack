// Go SUBPROCESS compute backend (Phase-2 spike). Spawns the plain-`go build` worker (no cgo, so
// no C toolchain needed — this sidesteps the broken local gcc/cgo/FFI path) and exchanges flat
// Float64 buffers over stdio. It is ASYNC (a process boundary), so it does NOT implement the sync
// ComputeBackend interface — that is the key spike finding: without a working in-process FFI, the
// owned-Go path costs async coloring + IPC serialization. Kept as the reference for when the
// toolchain is fixed (then a sync FFI backend replaces this).

import type { Subprocess } from "bun";

export class GoBackend {
  private Proc: Subprocess<"pipe", "pipe", "inherit">;
  private Reader: ReadableStreamDefaultReader<Uint8Array>;
  private Leftover: Uint8Array = new Uint8Array(0);
  // Serializes calls onto a single promise chain: MatMul reads/writes this.Reader and
  // this.Leftover across await points, and the stdio pipe carries one request at a time,
  // so concurrent callers must queue rather than interleave.
  private Queue: Promise<unknown> = Promise.resolve();

  constructor(WorkerPath = "GoKernels/worker.exe") {
    this.Proc = Bun.spawn([WorkerPath], { stdin: "pipe", stdout: "pipe", stderr: "inherit" });
    // Bun augments the reader type with readMany; we only use read(), so narrow to the DOM type.
    this.Reader = this.Proc.stdout.getReader() as unknown as ReadableStreamDefaultReader<Uint8Array>;
  }

  async MatMul(A: Float64Array, B: Float64Array, M: number, K: number, N: number): Promise<Float64Array> {
    const Run = this.Queue.then(() => this.MatMulOnce(A, B, M, K, N));
    this.Queue = Run.catch(() => {});
    return Run;
  }

  private async MatMulOnce(A: Float64Array, B: Float64Array, M: number, K: number, N: number): Promise<Float64Array> {
    const Header = new DataView(new ArrayBuffer(12));
    Header.setInt32(0, M, true);
    Header.setInt32(4, K, true);
    Header.setInt32(8, N, true);
    const Sink = this.Proc.stdin;
    Sink.write(new Uint8Array(Header.buffer));
    Sink.write(new Uint8Array(A.buffer, A.byteOffset, A.byteLength));
    Sink.write(new Uint8Array(B.buffer, B.byteOffset, B.byteLength));
    await Sink.flush();

    const Bytes = await this.ReadExactly(M * N * 8);
    const Out = new Float64Array(M * N);
    new Uint8Array(Out.buffer).set(Bytes); // copy into an 8-aligned buffer
    return Out;
  }

  private async ReadExactly(Count: number): Promise<Uint8Array> {
    const Chunks: Uint8Array[] = [];
    let Have = 0;
    if (this.Leftover.length > 0) {
      Chunks.push(this.Leftover);
      Have = this.Leftover.length;
      this.Leftover = new Uint8Array(0);
    }
    while (Have < Count) {
      const { value: Value, done: Done } = await this.Reader.read();
      if (Done || Value === undefined) throw new Error("Go worker closed unexpectedly");
      Chunks.push(Value);
      Have += Value.length;
    }
    const All = new Uint8Array(Have);
    let Offset = 0;
    for (const C of Chunks) {
      All.set(C, Offset);
      Offset += C.length;
    }
    if (Have > Count) this.Leftover = All.slice(Count);
    return All.subarray(0, Count);
  }

  Close(): void {
    this.Proc.stdin.end();
  }
}
