import React, { useState } from 'react';
import { useDispatch, useSelector } from 'react-redux';
import { makeStyles, Theme } from '@material-ui/core/styles';
import Grid from '@material-ui/core/Grid';
import TextField from '@material-ui/core/TextField';
import SpotifyIcon from './SpotifyIcon';
import { useSpotifyAuthSlice } from './slice';
import { selectSpotifyAuth } from './slice/selectors';
import IconButton from '@material-ui/core/IconButton';
import CircularProgress from '@material-ui/core/CircularProgress';
import GREY from '@material-ui/core/colors/grey';
import { selectSourcesAvailable } from '../SearchInputs/slice/selectors';

const useStyles = makeStyles((theme: Theme) => ({
  margin: {
    margin: theme.spacing(1),
  },
  wrapper: {
    margin: theme.spacing(1),
    position: 'relative',
  },
  circularProgress: {
    color: GREY[500],
    position: 'absolute',
    top: 11,
    left: 11,
    zIndex: 1,
  },
}));

interface Props {
  setSpotifyResults: React.Dispatch<any>;
  setHasError: React.Dispatch<
    React.SetStateAction<{
      error: boolean;
      message: string;
    }>
  >;
}

export default function SearchButton(props: Props) {
  const [searchInput, setSearchInput] = useState('');
  const [searching, setSearching] = useState(false);
  const [progress, setProgress] = useState(false);
  const { accessToken } = useSelector(selectSpotifyAuth);
  const dispatch = useDispatch();
  const { actions } = useSpotifyAuthSlice();
  const numSourcesAvailable = useSelector(selectSourcesAvailable);
  const classes = useStyles();
  const controller = new AbortController();

  const initiateGetResult = async ev => {
    ev.preventDefault();
    if (searchInput.length > 0) {
      try {
        setSearching(true);
        const progressTimer = setTimeout(setProgress(true), 1000);
        const apiTimer = setTimeout(() => controller.abort(), 6000);
        const API_URL = `https://api.spotify.com/v1/search?query=${encodeURIComponent(
          searchInput,
        )}&type=track&limit=30`;
        const result = await fetch(API_URL, {
          method: 'GET',
          cache: 'default',
          headers: {
            Authorization: 'Bearer ' + accessToken,
            Accept: 'application/json',
            'Content-Type': 'application/json',
          },
          signal: controller.signal,
        });
        clearTimeout(progressTimer);
        clearTimeout(apiTimer);
        if (result.status !== 200) {
          throw new Error('Non-200 code received from Spotify');
        }
        result.json().then(data => {
          props.setSpotifyResults(data);
        });
      } catch (error) {
        console.log('error', error);
        dispatch(actions.logOut()); // Something went wrong; try logging out
        props.setHasError({
          error: true,
          message:
            "Couldn't reach Spotify. Sorry about that. You can trying authorizing your account again.",
        });
      } finally {
        setSearching(false);
        setProgress(false);
      }
    }
  };

  const handleSearchChange = ev => {
    ev.preventDefault();
    setSearchInput(ev.target.value);
  };

  return (
    <>
      <form
        className="spotify-search-form"
        noValidate
        autoComplete="off"
        onSubmit={ev => initiateGetResult(ev)}
      >
        <Grid container spacing={1} alignItems="flex-end">
          <Grid item>
            <TextField
              id="standard-search"
              fullWidth
              disabled={numSourcesAvailable < 1}
              label={
                numSourcesAvailable < 1
                  ? 'Clear a track to add more'
                  : 'Search on Spotify'
              }
              defaultValue={searchInput}
              type="search"
              onChange={ev => handleSearchChange(ev)}
            />
          </Grid>
          <Grid item>
            <IconButton
              color="primary"
              disabled={searching || numSourcesAvailable < 1}
              aria-label="search for a song on Spotify"
              onClick={ev => initiateGetResult(ev)}
            >
              <SpotifyIcon
                iconProps={{}}
                fillColor={numSourcesAvailable < 1 ? '#dadada' : undefined}
              />
              {progress && (
                <CircularProgress
                  size={27}
                  className={classes.circularProgress}
                />
              )}
            </IconButton>
          </Grid>
        </Grid>
      </form>
    </>
  );
}
