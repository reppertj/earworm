import { createSelector } from '@reduxjs/toolkit';

import { RootState } from 'types';
import { initialState, errorAdapter } from '.';

const selectSlice = (state: RootState) => state.error || initialState;

const defaultErrorSelectors = errorAdapter.getSelectors(selectSlice);

export const selectNumErrors = defaultErrorSelectors.selectTotal;

export const selectAllErrors = defaultErrorSelectors.selectAll;

export const selectError = createSelector([selectSlice], state => state);
