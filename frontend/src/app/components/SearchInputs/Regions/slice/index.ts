import { PayloadAction, createEntityAdapter } from '@reduxjs/toolkit';
import { createSlice } from 'utils/@reduxjs/toolkit';
import { useInjectReducer } from 'utils/redux-injectors';
import { Region, RegionsState } from './types';

import { audioSourcesActions } from '../../AudioInputs/slice';

export const regionsAdapter = createEntityAdapter<Region>();

// There has to be a better way to do this
export const MAX_REGIONS = 6;
export const ALL_REGION_COLORS_ALPHA = [
  'rgba(141, 170, 145, 0.5)',
  'rgba(255, 207, 0, 0.5)',
  'rgba(66, 75, 84, 0.5)',
  'rgba(203, 133, 137, 0.5)',
  'rgba(146, 20, 12, 0.5)',
  'rgba(75, 179, 253, 0.5)',
];
export const ALL_REGION_COLORS_SOLID = [
  'rgba(141, 170, 145, 1)',
  'rgba(255, 207, 0, 1)',
  'rgba(66, 75, 84, 1)',
  'rgba(203, 133, 137, 1)',
  'rgba(146, 20, 12, 1)',
  'rgba(75, 179, 253, 1)',
];

export const initialState: RegionsState = regionsAdapter.getInitialState();

const slice = createSlice({
  name: 'regions',
  initialState,
  reducers: {
    regionAddedIfRoom(state, action: PayloadAction<Region>) {
      if (state.ids.length < MAX_REGIONS) {
        regionsAdapter.addOne(state, action.payload);
      }
    },
    regionUpdated: regionsAdapter.updateOne,
    regionRemoved: regionsAdapter.removeOne,
    regionsRemoved: regionsAdapter.removeMany,
  },
  extraReducers: builder => {
    builder.addCase(audioSourcesActions.sourceRemoved, (state, action) => {
      const sourceId = action.payload;
      const regionIds = state.ids.filter((_, idx) => {
        return state.entities[idx]?.sourceId === sourceId;
      });
      regionsAdapter.removeMany(state, regionIds);
    });
  },
});

export const { actions: regionsActions } = slice;

export const useRegionsSlice = () => {
  useInjectReducer({ key: slice.name, reducer: slice.reducer });
  return { actions: slice.actions };
};

/**
 * Example Usage:
 *
 * export function MyComponentNeedingThisSlice() {
 *  const { actions } = useRegionsSlice();
 *
 *  const onButtonClick = (evt) => {
 *    dispatch(actions.someAction());
 *   };
 * }
 */
