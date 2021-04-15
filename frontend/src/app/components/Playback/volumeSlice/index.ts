import { PayloadAction } from '@reduxjs/toolkit';
import { createSlice } from 'utils/@reduxjs/toolkit';
import { useInjectReducer } from 'utils/redux-injectors';
import { VolumeState } from './types';

export const initialState: VolumeState = { value: 0.8 };

const slice = createSlice({
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
  useInjectReducer({ key: slice.name, reducer: slice.reducer });
  return { actions: slice.actions };
};

/**
 * Example Usage:
 *
 * export function MyComponentNeedingThisSlice() {
 *  const { actions } = useVolumeSlice();
 *
 *  const onButtonClick = (evt) => {
 *    dispatch(actions.someAction());
 *   };
 * }
 */
