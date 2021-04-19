import { createEntityAdapter } from '@reduxjs/toolkit';
import { createSlice } from 'utils/@reduxjs/toolkit';
import { embeddingsSaga } from './saga';
import { EmbeddingsState, MultipleEmbeddings } from './types';

export const embeddingsAdapter = createEntityAdapter<MultipleEmbeddings>();

export const initialState: EmbeddingsState = embeddingsAdapter.getInitialState();

const slice = createSlice({
  name: 'embeddings',
  initialState,
  reducers: {
    embeddingsAdded: embeddingsAdapter.addOne,
    embeddingsRemoved: embeddingsAdapter.removeOne,
    embeddingsAllRemoved: embeddingsAdapter.removeAll,
  },
});

export const { actions: embeddingsActions } = slice;

export const useEmbeddingsSlice = () => {
  return {
    actions: slice.actions,
    reducer: slice.reducer,
    saga: embeddingsSaga,
  };
};

export default slice;
