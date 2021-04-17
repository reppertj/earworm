import { createSelector } from '@reduxjs/toolkit';
import { RootState } from 'types';
import {
  initialState,
  MAX_REGIONS,
  ALL_REGION_COLORS_ALPHA,
  ALL_REGION_COLORS_SOLID,
  regionsAdapter,
} from '.';

const selectSlice = (state: RootState) => state.regions || initialState;

const defaultRegionsSelectors = regionsAdapter.getSelectors(selectSlice);

export const selectRegions = createSelector([selectSlice], state => state);

export const selectNumRegions = defaultRegionsSelectors.selectTotal;

export const selectAllRegions = defaultRegionsSelectors.selectAll;

export const selectRegionByID = defaultRegionsSelectors.selectById;

export const selectRegionIdToAlphaColorMap = createSelector(
  [defaultRegionsSelectors.selectIds],
  ids =>
    Object.fromEntries(
      ids.map((id, idx) => [id, ALL_REGION_COLORS_ALPHA[idx]]),
    ),
);

export const selectRegionIdToSolidColorMap = createSelector(
  [defaultRegionsSelectors.selectIds],
  ids =>
    Object.fromEntries(
      ids.map((id, idx) => [id, ALL_REGION_COLORS_SOLID[idx]]),
    ),
);

export const selectRegionIdToShowIdMap = createSelector(
  [defaultRegionsSelectors.selectIds],
  ids => Object.fromEntries(ids.map((id, idx) => [id, idx + 1])),
);

export const selectRegionsAvailable = createSelector(
  [selectNumRegions],
  numRegions => Math.max(0, MAX_REGIONS - numRegions),
);
