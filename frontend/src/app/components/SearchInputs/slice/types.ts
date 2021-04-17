/* --- STATE --- */
import { EntityState } from '@reduxjs/toolkit';

export interface Source {
  id: string;
  originalData: string;
  ready: boolean;
  channelData?: string;
  sampleRate?: number;
}

export type AudioSourcesState = EntityState<Source>;
