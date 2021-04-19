import ls from 'local-storage';
import { PayloadAction } from '@reduxjs/toolkit';
import { createSlice } from 'utils/@reduxjs/toolkit';
import { spotifyAuthSaga, getTokenExpiryFromLocalStorage } from './saga';
import { SpotifyAuthState } from './types';

(ls as any).backend(localStorage);

const { loggedIn, token, expiry } = getTokenExpiryFromLocalStorage();

export const initialState: SpotifyAuthState = {
  loggedIn: loggedIn,
  accessToken: token,
  tokenExpiryDate: expiry,
};

const slice = createSlice({
  name: 'spotifyAuth',
  initialState,
  reducers: {
    logIn(state, action: PayloadAction<{ token: string; expiry: number }>) {
      state.loggedIn = true;
      state.accessToken = action.payload.token;
      state.tokenExpiryDate = action.payload.expiry;
    },
    logOut(state) {
      state.loggedIn = false;
      state.accessToken = undefined;
      state.tokenExpiryDate = undefined;
    },
  },
});

export const { actions: spotifyAuthActions } = slice;

export const useSpotifyAuthSlice = () => {
  return {
    actions: slice.actions,
    reducer: slice.reducer,
    saga: spotifyAuthSaga,
  };
};

export default slice;
