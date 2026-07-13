// Minimal, dependency-free HTML → text for general-web ingestion (M6). Drops script/style, strips
// tags, decodes the common entities, and collapses whitespace. Good enough to turn a fetched page
// into inspectable text for the isolated Raw tier (this content is never training-eligible).

export function HtmlToText(Html: string): string {
  return Html
    .replace(/<script[\s\S]*?<\/script>/gi, " ")
    .replace(/<style[\s\S]*?<\/style>/gi, " ")
    .replace(/<[^>]+>/g, " ")
    .replace(/&nbsp;/gi, " ")
    .replace(/&amp;/gi, "&")
    .replace(/&lt;/gi, "<")
    .replace(/&gt;/gi, ">")
    .replace(/&quot;/gi, '"')
    .replace(/&#39;/gi, "'")
    .replace(/\s+/g, " ")
    .trim();
}
