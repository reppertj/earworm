import { combineReducers } from '@reduxjs/toolkit';

import { configureStore, getDefaultMiddleware } from '@reduxjs/toolkit';
import createSagaMiddleware from 'redux-saga';
import { spawn } from 'redux-saga/effects';

import volumeSlice from 'app/components/Playback/volumeSlice';

import spotifyAuthSlice from 'app/components/Spotify/slice';
import { spotifyAuthSaga } from 'app/components/Spotify/slice/saga';

import audioSourcesSlice from 'app/components/SearchInputs/AudioInputs/slice';
import { audioSourcesSaga } from 'app/components/SearchInputs/AudioInputs/slice/saga';

import regionsSlice from 'app/components/SearchInputs/Regions/slice';
import errorSlice from 'app/components/Error/slice';

import searchResultsSlice from 'app/components/SearchResults/slice';
import { searchResultsSaga } from 'app/components/SearchResults/slice/saga';

import embeddingsSlice from 'app/components/SearchInputs/SearchMeta/slice';
import { embeddingsSaga } from 'app/components/SearchInputs/SearchMeta/slice/saga';

const reducers = {
  volume: volumeSlice.reducer,
  searchResults: searchResultsSlice.reducer,
  spotifyAuth: spotifyAuthSlice.reducer,
  regions: regionsSlice.reducer,
  audioSources: audioSourcesSlice.reducer,
  error: errorSlice.reducer,
  embeddings: embeddingsSlice.reducer,
};

export function* rootSaga() {
  yield spawn(spotifyAuthSaga);
  yield spawn(audioSourcesSaga);
  yield spawn(searchResultsSaga);
  yield spawn(embeddingsSaga);
}

export function configureAppStore() {
  const reduxSagaMonitorOptions = {};
  const sagaMiddleware = createSagaMiddleware(reduxSagaMonitorOptions);

  const middlewares = [sagaMiddleware];

  const store = configureStore({
    reducer: combineReducers(reducers),
    middleware: [
      ...getDefaultMiddleware({
        immutableCheck: false,
        serializableCheck: false,
      }),
      ...middlewares,
    ],
    devTools: process.env.NODE_ENV !== 'production',
  });

  return { ...store, runSaga: sagaMiddleware.run };
}
