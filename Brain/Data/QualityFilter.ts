// Heuristic code-quality filter (Phase 3, phi-1-style "quality is the real lever"). Cheap,
// deterministic signals that keep clean, human-written code and drop minified/generated/binary-ish
// junk. Not perfect — a first-pass filter before the corpus reaches training.

export type QualityResult = { Score: number; Passed: boolean; Reasons: string[] };

export type QualityOptions = {
  MaxAvgLineLength?: number; // very long avg lines => minified/generated
  MaxLongLineFraction?: number; // fraction of lines over 400 chars
  MinPrintableFraction?: number; // guards against binary/base64 blobs
  MinAlphaFraction?: number; // guards against near-empty / symbol soup
  Threshold?: number; // pass if Score >= Threshold
};

export function ScoreCodeQuality(Text: string, Options: QualityOptions = {}): QualityResult {
  const MaxAvgLineLength = Options.MaxAvgLineLength ?? 120;
  const MaxLongLineFraction = Options.MaxLongLineFraction ?? 0.1;
  const MinPrintableFraction = Options.MinPrintableFraction ?? 0.95;
  const MinAlphaFraction = Options.MinAlphaFraction ?? 0.25;
  const Threshold = Options.Threshold ?? 0.6;

  const Reasons: string[] = [];
  let Score = 1;

  const Lines = Text.split("\n");
  const NonEmpty = Lines.filter((L) => L.trim().length > 0);
  if (NonEmpty.length === 0) {
    return { Score: 0, Passed: false, Reasons: ["empty"] };
  }

  const AvgLineLength = Text.length / Lines.length;
  if (AvgLineLength > MaxAvgLineLength) {
    Score -= 0.4;
    Reasons.push(`avg line length ${AvgLineLength.toFixed(0)} > ${MaxAvgLineLength} (minified?)`);
  }

  const LongLines = Lines.filter((L) => L.length > 400).length;
  const LongFraction = LongLines / Lines.length;
  if (LongFraction > MaxLongLineFraction) {
    Score -= 0.3;
    Reasons.push(`long-line fraction ${LongFraction.toFixed(2)} > ${MaxLongLineFraction}`);
  }

  let Printable = 0;
  let Alpha = 0;
  for (let I = 0; I < Text.length; I++) {
    const Code = Text.charCodeAt(I);
    // Non-ASCII (Code >= 128) is legitimate TEXT — Arabic, other scripts, emoji, unicode identifiers —
    // NOT binary. Counting it as printable+alpha keeps the filter from wrongly rejecting non-English
    // prose as "binary"; real binary is control chars (< 32, excluding tab/newline/CR), still caught.
    if (Code === 9 || Code === 10 || Code === 13 || (Code >= 32 && Code < 127) || Code >= 128) Printable++;
    if ((Code >= 65 && Code <= 90) || (Code >= 97 && Code <= 122) || Code >= 128) Alpha++;
  }
  const PrintableFraction = Printable / Text.length;
  const AlphaFraction = Alpha / Text.length;
  if (PrintableFraction < MinPrintableFraction) {
    Score -= 0.5;
    Reasons.push(`printable fraction ${PrintableFraction.toFixed(2)} < ${MinPrintableFraction} (binary/base64?)`);
  }
  if (AlphaFraction < MinAlphaFraction) {
    Score -= 0.2;
    Reasons.push(`alpha fraction ${AlphaFraction.toFixed(2)} < ${MinAlphaFraction}`);
  }

  if (Score < 0) Score = 0;
  return { Score, Passed: Score >= Threshold, Reasons };
}
