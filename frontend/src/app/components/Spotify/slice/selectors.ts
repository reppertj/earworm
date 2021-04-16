import { createSelector } from '@reduxjs/toolkit';

import { RootState } from 'types';
import { initialState } from '.';

const selectSlice = (state: RootState) => state.spotifyAuth || initialState;

export const selectSpotifyAuth = createSelector([selectSlice], state => state);
