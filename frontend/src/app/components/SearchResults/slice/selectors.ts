import { createSelector } from '@reduxjs/toolkit';

import { RootState } from 'types';
import { initialState, searchResultsEntityAdapter } from '.';

const selectSlice = (state: RootState) => state.searchResults || initialState;

const defaultSearchResultsSelectors = searchResultsEntityAdapter.getSelectors(
  selectSlice,
);

export const selectAllSearchResults = defaultSearchResultsSelectors.selectAll;

export const selectSearchResults = createSelector(
  [selectSlice],
  state => state,
);
