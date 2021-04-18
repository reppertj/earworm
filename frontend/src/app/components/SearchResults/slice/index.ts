import { PayloadAction, createEntityAdapter } from '@reduxjs/toolkit';
import { createSlice } from 'utils/@reduxjs/toolkit';
import { useInjectReducer, useInjectSaga } from 'utils/redux-injectors';
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
  useInjectReducer({ key: slice.name, reducer: slice.reducer });
  useInjectSaga({ key: slice.name, saga: searchResultsSaga });
  return { actions: slice.actions };
};

/**
 * Example Usage:
 *
 * export function MyComponentNeedingThisSlice() {
 *  const { actions } = useSearchResultsSlice();
 *
 *  const onButtonClick = (evt) => {
 *    dispatch(actions.someAction());
 *   };
 * }
 */
