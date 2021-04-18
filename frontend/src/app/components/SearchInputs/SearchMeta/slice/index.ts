import { PayloadAction, createEntityAdapter } from '@reduxjs/toolkit';
import { createSlice } from 'utils/@reduxjs/toolkit';
import { useInjectReducer, useInjectSaga } from 'utils/redux-injectors';
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
  useInjectReducer({ key: slice.name, reducer: slice.reducer });
  useInjectSaga({ key: slice.name, saga: embeddingsSaga });
  return { actions: slice.actions };
};

/**
 * Example Usage:
 *
 * export function MyComponentNeedingThisSlice() {
 *  const { actions } = useEmbeddingsSlice();
 *
 *  const onButtonClick = (evt) => {
 *    dispatch(actions.someAction());
 *   };
 * }
 */
