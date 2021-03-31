import React, { useEffect } from 'react';
import {
  getHashParams,
  removeHashParamsFromUrl,
  getAuthorizeHref,
} from '../utils/apiUtils';
import ls from 'local-storage';

const authEndpoint = 'https://accounts.spotify.com/authorize';

const scopes = [];

interface Props {
  loggedIn: boolean;
  setLoggedIn: React.Dispatch<React.SetStateAction<boolean>>;
  setAccessToken: React.Dispatch<React.SetStateAction<string | null>>;
  setTokenExpiryDate: React.Dispatch<React.SetStateAction<number | null>>;
  setHasError: React.Dispatch<
    React.SetStateAction<{
      error: boolean;
      message: string;
    }>
  >;
}

export default function SpotifyAuthorizationButton(props: Props) {
  (ls as any).backend(localStorage);
  console.log(ls);
  useEffect(() => {
    const params = getHashParams();
    const accessToken = params.access_token;
    const EXPIRY_MULT = (1000 * 9) / 10;
    const expiresIn = Number(params.expires_in) * EXPIRY_MULT + Date.now();
    try {
      if (accessToken || ls('accessToken')) {
        props.setLoggedIn(true);
        ls('SpotifyAccessToken', accessToken);
        ls('SpotifyExpiry', expiresIn);
        props.setAccessToken(accessToken);
        props.setTokenExpiryDate(expiresIn);
        removeHashParamsFromUrl();
      } else if (
        ((ls('SpotifyAccessToken') as unknown) as string) &&
        Date.now() < Number(ls('SpotifyExpiry') as unknown) &&
        props.loggedIn
      ) {
        props.setAccessToken((ls('SpotifyAccessToken') as unknown) as string);
        props.setTokenExpiryDate(
          Number((ls('SpotifyExpiry') as unknown) as string),
        );
      }
    } catch {
      props.setHasError({
        error: true,
        message: "Couldn't connect to your Spotify account. Sorry about that.",
      });
      props.setLoggedIn(false);
      props.setAccessToken('');
      ls('accessToken', '');
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
        <button
          aria-label="Authorize Spotify account using OAuth 2.0"
          onClick={ev => handleRedirect(ev)}
        >
          or use your Spotify account
        </button>
      }
    </div>
  );
}
