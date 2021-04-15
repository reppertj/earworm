import { createSelector } from '@reduxjs/toolkit';

import { RootState } from 'types';
import { initialState } from '.';

const selectSlice = (state: RootState) => state.volumeSlice || initialState;

export const selectVolumeSlice = createSelector([selectSlice], state => state);
