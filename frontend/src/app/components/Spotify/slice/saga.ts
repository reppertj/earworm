import { take, call } from 'redux-saga/effects';
import { spotifyAuthActions as actions } from '.';

import ls from 'local-storage';

export function getTokenExpiryFromLocalStorage() {
  (ls as any).backend(localStorage);
  const token = ls('SpotifyAccessToken') as string | undefined;
  const expiry = ls('SpotifyExpiry') as string | undefined;
  if (expiry) {
    const expiryNum = Number(expiry);
    if (Date.now() < expiryNum && token) {
      return { loggedIn: true, token, expiry: expiryNum };
    }
  }
  return { loggedIn: false, token: undefined, expiry: undefined };
}

function assignTokenExpiryToLocalStorage({ payload }) {
  if (payload.token && payload.expiry) {
    (ls as any).backend(localStorage);
    ls('SpotifyAccessToken', payload.token);
    ls('SpotifyExpiry', payload.expiry);
  }
}

export function* spotifyAuthSaga() {
  const action = yield take(actions.logIn.type);
  yield call(assignTokenExpiryToLocalStorage, action);
}
