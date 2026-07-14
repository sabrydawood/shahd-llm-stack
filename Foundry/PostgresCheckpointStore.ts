// Store model checkpoints (weights + optimizer + RNG + config + tokenizer) in Postgres, so a trained
// model is durable and synced alongside the corpus + chat — not a gitignored file a `git clean` can
// wipe. Brain stays Postgres-agnostic: it builds/parses the checkpoint object (BuildCheckpoint /
// ParseCheckpoint); this Foundry store just persists that object's JSON as a row. Lightweight meta
// (params/arch/vocab) is stored separately so List() never has to fetch the multi-MB payload.
//
// NOTE: fine for the current small models (~15MB JSON). For future GB-scale models, move the payload
// to object storage and keep only the metadata row here.

import postgres from "postgres";
import type { Checkpoint } from "../Brain/Checkpoint/CheckpointFormat.ts";
import { ParseCheckpoint } from "../Brain/Checkpoint/CheckpointReader.ts";

export type CheckpointSummary = { Name: string; CreatedAt: string; Params: number; Vocab: number; Arch: string; Corpus: string };

type MetaRow = { name: string; created_at: string; meta: string };
type DataRow = { data: string };

function MetaOf(Ckpt: Checkpoint): { params: number; vocab: number; arch: string; corpus: string } {
  return {
    params: Ckpt.Params.reduce((Acc, P) => Acc + P.Rows * P.Cols, 0),
    vocab: Ckpt.Config.Model.VocabSize,
    arch: `emb${Ckpt.Config.Model.EmbedDim} L${Ckpt.Config.Model.NumLayers} ctx${Ckpt.Config.Model.BlockSize}`,
    corpus: String((Ckpt.Meta as Record<string, unknown>)["Corpus"] ?? ""),
  };
}

export class PostgresCheckpointStore {
  private Sql: ReturnType<typeof postgres>;
  private Ready: Promise<void>;

  constructor(Url: string) {
    this.Sql = postgres(Url);
    this.Ready = this.Migrate().catch((Caught) => {
      console.warn(`PostgresCheckpointStore: migration deferred: ${(Caught as Error).message}`);
    });
  }

  private async Migrate(): Promise<void> {
    await this.Sql`CREATE TABLE IF NOT EXISTS checkpoints (name TEXT PRIMARY KEY, format_version INT NOT NULL, data TEXT NOT NULL, meta TEXT NOT NULL, created_at TEXT NOT NULL)`;
  }

  async Save(Name: string, Ckpt: Checkpoint, CreatedAt: string): Promise<void> {
    await this.Ready;
    const Data = JSON.stringify(Ckpt);
    const Meta = JSON.stringify(MetaOf(Ckpt));
    await this.Sql`
      INSERT INTO checkpoints (name, format_version, data, meta, created_at)
      VALUES (${Name}, ${Ckpt.FormatVersion}, ${Data}, ${Meta}, ${CreatedAt})
      ON CONFLICT (name) DO UPDATE SET format_version = EXCLUDED.format_version, data = EXCLUDED.data, meta = EXCLUDED.meta, created_at = EXCLUDED.created_at`;
  }

  async Load(Name: string): Promise<Checkpoint | null> {
    await this.Ready;
    const Rows = (await this.Sql`SELECT data FROM checkpoints WHERE name = ${Name} LIMIT 1`) as unknown as DataRow[];
    return Rows[0] ? ParseCheckpoint(Rows[0].data) : null;
  }

  async List(): Promise<CheckpointSummary[]> {
    await this.Ready;
    const Rows = (await this.Sql`SELECT name, created_at, meta FROM checkpoints ORDER BY created_at DESC`) as unknown as MetaRow[];
    return Rows.map((R) => {
      const M = JSON.parse(R.meta) as { params: number; vocab: number; arch: string; corpus: string };
      return { Name: R.name, CreatedAt: R.created_at, Params: M.params, Vocab: M.vocab, Arch: M.arch, Corpus: M.corpus };
    });
  }

  async Close(): Promise<void> {
    await this.Sql.end();
  }
}
