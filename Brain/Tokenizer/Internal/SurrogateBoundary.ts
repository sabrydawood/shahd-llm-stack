// Shared boundary safety for any code that slices a JS string at a computed index (chunk splitting,
// FIM cut points, ...). JS strings are UTF-16: an astral character (emoji, many CJK/math symbols) is
// stored as a high surrogate (0xD800-0xDBFF) followed by a low surrogate (0xDC00-0xDFFF). Slicing
// between the two yields two lone surrogates — an unpaired surrogate corrupts the string (invalid
// UTF-16, mangled character, lossy round-trip through code points / UTF-8 re-encoding).

const HighSurrogateStart = 0xd800;
const HighSurrogateEnd = 0xdbff;
const LowSurrogateStart = 0xdc00;
const LowSurrogateEnd = 0xdfff;

/** If Idx falls between a high surrogate (at Idx-1) and a low surrogate (at Idx), shift it past the
 *  pair by 1 so a .slice() at the returned index never splits one astral character in two. */
export function SafeBoundary(Text: string, Idx: number): number {
  if (Idx <= 0 || Idx >= Text.length) return Idx;
  const Prev = Text.charCodeAt(Idx - 1);
  const Next = Text.charCodeAt(Idx);
  const PrevIsHigh = Prev >= HighSurrogateStart && Prev <= HighSurrogateEnd;
  const NextIsLow = Next >= LowSurrogateStart && Next <= LowSurrogateEnd;
  return PrevIsHigh && NextIsLow ? Idx + 1 : Idx;
}
