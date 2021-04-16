/**
 *
 * SpotifyAuthorizationButton
 *
 */
import React, { useEffect } from 'react';
import { useDispatch } from 'react-redux';
import { useSpotifyAuthSlice } from '../slice';
import Button from '@material-ui/core/Button';
import useTheme from '@material-ui/core/styles/useTheme';

import SpotifyIcon from '../../SpotifyIcon';

import {
  getHashParams,
  removeHashParamsFromUrl,
  getAuthorizeHref,
} from '../../../utils/spotify';
import ls from 'local-storage';

const authEndpoint = 'https://accounts.spotify.com/authorize';

const scopes = [];

interface Props {
  setHasError: React.Dispatch<
    React.SetStateAction<{
      error: boolean;
      message: string;
    }>
  >;
}
export function SpotifyAuthorizationButton(props: Props) {
  const theme = useTheme();
  const dispatch = useDispatch();
  const { actions } = useSpotifyAuthSlice();
  // const { }
  console.log(ls);
  useEffect(() => {
    const params = getHashParams();
    const token = params.access_token;
    const EXPIRY_MULT = (1000 * 9) / 10;
    const expiry = Number(params.expires_in) * EXPIRY_MULT + Date.now();
    try {
      if (token) {
        dispatch(actions.logIn({ token, expiry }));
        removeHashParamsFromUrl();
      }
    } catch (e) {
      props.setHasError({
        error: true,
        message: "Couldn't connect to your Spotify account. Sorry about that.",
      });
      dispatch(actions.logOut());
    }

    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  const handleRedirect = ev => {
    ev.preventDefault();
    let href = getAuthorizeHref(authEndpoint, scopes);
    window.location.href = href;
  };

  return (
    <div>
      {
        <Button
          variant="contained"
          color="primary"
          aria-label="Authorize Spotify account using OAuth 2.0"
          onClick={ev => handleRedirect(ev)}
        >
          or connect your Spotify&nbsp;
          <SpotifyIcon
            iconProps={{ fontSize: 'small' }}
            fillColor={theme.palette.primary.contrastText}
          />
          &nbsp;account
        </Button>
      }
    </div>
  );
}
