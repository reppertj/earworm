import { createEntityAdapter } from '@reduxjs/toolkit';
import { createSlice } from 'utils/@reduxjs/toolkit';
import { useInjectReducer } from 'utils/redux-injectors';
import { Error, ErrorState } from './types';

export const errorAdapter = createEntityAdapter<Error>();

export const initialState: ErrorState = errorAdapter.getInitialState();

const slice = createSlice({
  name: 'error',
  initialState,
  reducers: {
    errorAdded: errorAdapter.addOne,
    errorRemoved: errorAdapter.removeOne,
  },
});

export const { actions: errorActions } = slice;

export const useErrorSlice = () => {
  useInjectReducer({ key: slice.name, reducer: slice.reducer });
  return { actions: slice.actions };
};

/**
 * Example Usage:
 *
 * export function MyComponentNeedingThisSlice() {
 *  const { actions } = useErrorSlice();
 *
 *  const onButtonClick = (evt) => {
 *    dispatch(actions.someAction());
 *   };
 * }
 */
