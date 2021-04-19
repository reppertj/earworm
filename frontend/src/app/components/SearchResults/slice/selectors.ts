import { createSelector } from '@reduxjs/toolkit';

import { RootState } from 'types';
import { initialState, searchResultsEntityAdapter } from '.';

const selectSlice = (state: RootState) => state.searchResults || initialState;

const defaultSearchResultsSelectors = searchResultsEntityAdapter.getSelectors(
  selectSlice,
);

export const selectAllSearchResults = defaultSearchResultsSelectors.selectAll;

export const selectSearchResultsIds = defaultSearchResultsSelectors.selectIds;

export const selectResultsExist = createSelector(
  [selectAllSearchResults],
  results => results.length > 0,
);

export const selectSearchResults = createSelector(
  [selectSlice],
  state => state,
);
