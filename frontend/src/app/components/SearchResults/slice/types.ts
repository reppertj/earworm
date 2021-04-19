import { EntityState } from '@reduxjs/toolkit';
/* --- STATE --- */

export interface SearchResult {
  title: {
    name: string;
    url: string;
  };
  artist: string;
  license: {
    name: string;
    url: string;
  };
  provider: {
    name: string;
    url: string;
  };
  previewUrl: string | undefined;
  percentMatch: number;
}

export type SearchResultsState = EntityState<SearchResult>;
