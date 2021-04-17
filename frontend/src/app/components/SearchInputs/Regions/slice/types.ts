import { EntityState } from '@reduxjs/toolkit';
/* --- STATE --- */
export interface Region {
  id: string;
  sourceId: string;
  seconds: number;
  preference: number;
}

export type RegionsState = EntityState<Region>;
