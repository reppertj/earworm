import { EntityState } from '@reduxjs/toolkit';
/* --- STATE --- */

export interface MultipleEmbeddings {
  id: string;
  embeddings: number[][];
}

export type EmbeddingsState = EntityState<MultipleEmbeddings>;
