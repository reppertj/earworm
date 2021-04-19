import { PayloadAction } from '@reduxjs/toolkit';
import { createSlice } from 'utils/@reduxjs/toolkit';
import { VolumeState } from './types';

export const initialState: VolumeState = { value: 0.8 };

export const slice = createSlice({
  name: 'volume',
  initialState,
  reducers: {
    changeVolume(state, action: PayloadAction<number>) {
      state.value = action.payload;
    },
  },
});

export const { actions: volumeActions } = slice;

export const useVolumeSlice = () => {
  return { actions: slice.actions, reducer: slice.reducer };
};

export default slice;
