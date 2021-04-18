import { EntityState } from '@reduxjs/toolkit';
/* --- STATE --- */

export interface SearchResult {
  title: string;
  artist: string;
  url: string;
  licenseName: string;
  licenseUrl: string;
  providerName: string;
  providerUrl: string;
  previewUrl: string | undefined;
  percentMatch: number;
}

export type SearchResultsState = EntityState<SearchResult>;
