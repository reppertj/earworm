/**
 * Create the store with dynamic reducers
 */

import {
  configureStore,
  getDefaultMiddleware,
  Action,
  Update,
} from '@reduxjs/toolkit';
import { Source } from '../app/components/SearchInputs/slice/types';
import { createInjectorsEnhancer } from 'redux-injectors';
import createSagaMiddleware from 'redux-saga';

import { createReducer } from './reducers';

export function configureAppStore() {
  const reduxSagaMonitorOptions = {};
  const sagaMiddleware = createSagaMiddleware(reduxSagaMonitorOptions);
  const { run: runSaga } = sagaMiddleware;

  // Create the store with saga middleware
  const middlewares = [sagaMiddleware];

  const enhancers = [
    createInjectorsEnhancer({
      createReducer,
      runSaga,
    }),
  ];

  interface ActionWithPayload<T> extends Action {
    payload: T;
  }

  const store = configureStore({
    reducer: createReducer(),
    middleware: [
      ...getDefaultMiddleware({
        immutableCheck: false,
        serializableCheck: false,
      }),
      ...middlewares,
    ],
    devTools: false || process.env.NODE_ENV !== 'production',
    enhancers,
  });

  return store;
}
