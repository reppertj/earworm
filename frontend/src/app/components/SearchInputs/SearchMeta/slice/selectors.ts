import { createSelector } from '@reduxjs/toolkit';

import { RootState } from 'types';
import { embeddingsAdapter, initialState } from '.';

const selectSlice = (state: RootState) => state.embeddings || initialState;
const defaultEmeddingsSelectors = embeddingsAdapter.getSelectors(selectSlice);

export const selectAllEmbeddings = defaultEmeddingsSelectors.selectAll;
export const selectNumEmbeddings = defaultEmeddingsSelectors.selectTotal;
export const selectEmbeddingsById = defaultEmeddingsSelectors.selectById;

export const selectEmbeddings = createSelector([selectSlice], state => state);
