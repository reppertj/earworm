import { createEntityAdapter } from '@reduxjs/toolkit';
import { createSlice } from 'utils/@reduxjs/toolkit';
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
  return { actions: slice.actions, reducer: slice.reducer };
};

export default slice;
