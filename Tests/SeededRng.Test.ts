import { test, expect } from "bun:test";
import { CreateRngStreams, SeededRng } from "../Brain/Random/SeededRng.ts";

test("SeededRng is reproducible and stays in [0, 1)", () => {
  const A = new SeededRng(42);
  const B = new SeededRng(42);
  for (let I = 0; I < 100; I++) {
    const X = A.NextFloat();
    expect(X).toBe(B.NextFloat());
    expect(X).toBeGreaterThanOrEqual(0);
    expect(X).toBeLessThan(1);
  }
});

test("named streams are independent (ablation-contamination fix)", () => {
  const First = CreateRngStreams(7);
  for (let I = 0; I < 1000; I++) First.InitRng.NextFloat(); // heavy InitRng usage
  const FirstData = [First.DataRng.NextFloat(), First.DataRng.NextFloat(), First.DataRng.NextFloat()];

  const Second = CreateRngStreams(7);
  for (let I = 0; I < 5; I++) Second.InitRng.NextFloat(); // different InitRng usage
  const SecondData = [Second.DataRng.NextFloat(), Second.DataRng.NextFloat(), Second.DataRng.NextFloat()];

  expect(SecondData).toEqual(FirstData); // DataRng sequence must be unaffected by InitRng draws
});

test("distinct streams produce distinct sequences", () => {
  const S = CreateRngStreams(7);
  expect(S.InitRng.NextFloat()).not.toBe(S.DataRng.NextFloat());
});
