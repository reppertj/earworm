import { PayloadAction, createEntityAdapter, Update } from '@reduxjs/toolkit';
import { createSlice } from 'utils/@reduxjs/toolkit';
import { AudioSourcesState, Source } from './types';

export const MAX_SOURCES = 3;

export const sourcesAdapter = createEntityAdapter<Source>();

export const initialState: AudioSourcesState = sourcesAdapter.getInitialState();

export const slice = createSlice({
  name: 'audioSources',
  initialState,
  reducers: {
    sourcesAddedIfRoom(state, action: PayloadAction<Source[]>) {
      const numRemaining = Math.max(0, MAX_SOURCES - state.ids.length);
      const sourcesToAdd = action.payload.slice(0, numRemaining);
      sourcesAdapter.addMany(state as AudioSourcesState, sourcesToAdd);
    },
    sourceRemoved(state, action: PayloadAction<string>) {
      const sourceId = action.payload;
      sourcesAdapter.removeOne(state as AudioSourcesState, sourceId);
    },
    sourceUpdated(state, action: PayloadAction<Update<Source>>) {
      sourcesAdapter.updateOne(state as AudioSourcesState, action);
    },
  },
});

export const { actions: audioSourcesActions } = slice;

export const useAudioSourcesSlice = () => {
  return { actions: slice.actions };
};

export default slice;
