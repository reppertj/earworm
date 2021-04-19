import { createEntityAdapter } from '@reduxjs/toolkit';
import { createSlice } from 'utils/@reduxjs/toolkit';
import { searchResultsSaga } from './saga';
import { SearchResult, SearchResultsState } from './types';

export const searchResultsEntityAdapter = createEntityAdapter<SearchResult>();
export const initialState: SearchResultsState = searchResultsEntityAdapter.getInitialState();

const slice = createSlice({
  name: 'searchResults',
  initialState,
  reducers: {
    searchResultsAllSet: searchResultsEntityAdapter.setAll,
    searchResultsAllRemoved: searchResultsEntityAdapter.removeAll,
  },
});

export const { actions: searchResultsActions } = slice;

export const useSearchResultsSlice = () => {
  return {
    actions: slice.actions,
    reducer: slice.reducer,
    saga: searchResultsSaga,
  };
};

export default slice;
