import type Database from "better-sqlite3";

export interface Chunk {
  file: string;
  start: number;
  text: string;
  hash: string;
}

export interface RagDb {
  db: Database.Database;
  embedDim: number;
  embedModel: string;
}

export interface Ranked {
  chunkId: number;
  score: number;
}
