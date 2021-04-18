/* --- STATE --- */
import { EntityState } from '@reduxjs/toolkit';

export interface Source {
  id: string;
  originalData: string;
  ready: boolean;
  channelData?: Float32Array[];
  sampleRate?: number;
}

export type AudioSourcesState = EntityState<Source>;
