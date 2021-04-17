import { createSelector } from '@reduxjs/toolkit';

import { RootState } from 'types';
import { initialState } from '.';

const selectSlice = (state: RootState) => state.error || initialState;

export const selectError = createSelector([selectSlice], state => state);
