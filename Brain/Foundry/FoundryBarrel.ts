// Public surface of the Data Foundry (M3) — the tiered, inspectable, owned dataset layer.

export type { DocumentRecord, Tier, Origin } from "./DocumentRecord.ts";
export { ClassifyDocument } from "./Tiering.ts";
export type { TierDecision } from "./Tiering.ts";
export { HashingEmbedding, CosineSimilarity } from "./Embedding.ts";
export type { DocumentStore, SimilarHit } from "./DocumentStore.ts";
export { InMemoryDocumentStore } from "./InMemoryDocumentStore.ts";
export { BuildReport, RenderReportText } from "./QualityReport.ts";
export type { FoundryReport } from "./QualityReport.ts";
export { IngestDocuments, ExportTrainingText } from "./Ingest.ts";
export type { SourceInput, IngestStats } from "./Ingest.ts";
export { CreateDashboardHandler, StartDashboard } from "./Dashboard.ts";
export type { DashboardHandler } from "./Dashboard.ts";
export { HtmlToText } from "./HtmlToText.ts";
export { IngestFromWeb } from "./WebSource.ts";
export type { WebProvider } from "./WebSource.ts";
export { CreateGitHubProvider } from "./GitHubProvider.ts";
export type { GitHubOptions, HttpJson } from "./GitHubProvider.ts";
export { CreateWebSearchProvider } from "./WebSearchProvider.ts";
export type { SearchBackend, SearchHit, PageFetch, WebSearchOptions } from "./WebSearchProvider.ts";
