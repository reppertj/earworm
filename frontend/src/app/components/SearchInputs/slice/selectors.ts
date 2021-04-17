import { createSelector } from '@reduxjs/toolkit';

import { RootState } from 'types';
import { initialState, sourcesAdapter, MAX_SOURCES } from '.';

const selectSlice = (state: RootState) => state.audioSources || initialState;

const defaultSourceSelectors = sourcesAdapter.getSelectors(selectSlice);

export const selectAudioSources = createSelector([selectSlice], state => state);

export const selectNumSources = defaultSourceSelectors.selectTotal;

export const selectSourceByID = defaultSourceSelectors.selectById;

export const selectAllSources = defaultSourceSelectors.selectAll;

export const selectAllEntities = defaultSourceSelectors.selectEntities;

export const selectSourcesAvailable = createSelector(
  [selectNumSources],
  numSources => MAX_SOURCES - numSources,
);
