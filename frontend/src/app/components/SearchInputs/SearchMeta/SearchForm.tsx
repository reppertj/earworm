import React, { useCallback, useEffect, useMemo, useState } from 'react';
import { useDispatch, useSelector } from 'react-redux';
import {
  selectAllEntities,
  selectAllSources,
  selectNumSources,
  selectSourcesAvailable,
} from '../AudioInputs/slice/selectors';
import {
  selectAllRegions,
  selectNumRegions,
  selectRegionsAvailable,
} from '../Regions/slice/selectors';
import { useErrorSlice } from '../../Error/slice';

import Box from '@material-ui/core/Box';
import Container from '@material-ui/core/Container';
import makeStyles from '@material-ui/core/styles/makeStyles';
import Grid from '@material-ui/core/Grid';
import Typography from '@material-ui/core/Typography';
import Button from '@material-ui/core/Button';
import CloudUploadIcon from '@material-ui/icons/CloudUpload';
import Tooltip from '@material-ui/core/Tooltip';

import { useDropzone } from 'react-dropzone';
import { ConditionalWrapper } from '../../ConditionalWrapper';
import { UploadSurferPair } from '../AudioInputs/UploadSurfer';
import SpotifySearchForm from '../../Spotify/SpotifySearchForm';
import SearchProgress from './SearchProgress';
import { SpotifyAuthorizationButton } from '../../Spotify/SpotifyAuthorizationButton';
import { SpotifyChooser } from '../../Spotify/SpotifyResultChooser';
import Worker from '../../../worker';
import { selectSpotifyAuth } from '../../Spotify/slice/selectors';
import { useAudioSourcesSlice } from '../AudioInputs/slice';
import { Source } from '../AudioInputs/slice/types';
import { nanoid } from '@reduxjs/toolkit';
import { embeddingsActions, useEmbeddingsSlice } from './slice';

var inferenceWorker = new Worker();

const useStyles = makeStyles(theme => ({
  searchForm: {
    padding: theme.spacing(2),
  },
}));

function MusicSearchForm(props) {
  const [hasError, setHasError] = useState({ error: false, message: '' });
  const [spotifyResults, setSpotifyResults] = useState<any>(null);
  const [chooserOpen, setChooserOpen] = useState(false);
  const dispatch = useDispatch();
  const { actions: errorActions } = useErrorSlice();
  const { loggedIn, tokenExpiryDate } = useSelector(selectSpotifyAuth);

  const SpotifyElement = useMemo(() => {
    if (
      !loggedIn ||
      hasError.error ||
      (tokenExpiryDate && tokenExpiryDate < Date.now())
    ) {
      return <SpotifyAuthorizationButton setHasError={setHasError} />;
    } else {
      return (
        <SpotifySearchForm
          setHasError={setHasError}
          setSpotifyResults={setSpotifyResults}
        />
      );
    }
  }, [loggedIn, tokenExpiryDate, hasError]);

  useEffect(() => {
    if (hasError.error) {
      dispatch(
        errorActions.errorAdded({
          id: nanoid(),
          error: true,
          message:
            hasError.message ||
            'Encountered an error. Try refreshing the page.',
        }),
      );
      setHasError({ error: false, message: '' });
    }
  }, [dispatch, errorActions, hasError]);

  useEffect(() => {
    if (!chooserOpen && spotifyResults) {
      if (!spotifyResults.tracks) {
        dispatch(
          errorActions.errorAdded({
            id: nanoid(),
            error: true,
            message:
              "Couldn't process the results from Spotify. Sorry about that.",
          }),
        );
      } else if (spotifyResults.tracks.items.length < 1) {
        dispatch(
          errorActions.errorAdded({
            id: nanoid(),
            error: true,
            message: 'No results! Try searching for something else.',
          }),
        );
      } else {
        setChooserOpen(true);
      }
    }
  }, [spotifyResults, chooserOpen, dispatch, errorActions]);

  return (
    <>
      <UploadWindow SpotifyElement={SpotifyElement} />

      {chooserOpen && (
        <SpotifyChooser
          spotifyResults={spotifyResults}
          setSpotifyResults={setSpotifyResults}
          open={chooserOpen}
          setOpen={setChooserOpen}
        />
      )}
    </>
  );
}

interface UploadWindowProps {
  SpotifyElement: JSX.Element;
}

function UploadWindow(props: UploadWindowProps) {
  const dispatch = useDispatch();
  const { actions: errorActions } = useErrorSlice();
  const { actions: sourceActions } = useAudioSourcesSlice();
  const numSourcesAvailable = useSelector(selectSourcesAvailable);
  const numSources = useSelector(selectNumSources);

  const { open, getRootProps, getInputProps } = useDropzone({
    onDrop: acceptedFiles => {
      try {
        const newSources = acceptedFiles.map(file => {
          const objectUrl = URL.createObjectURL(file);
          const source: Source = {
            id: nanoid(),
            originalData: objectUrl,
            ready: false,
          };
          return source;
        });
        dispatch(sourceActions.sourcesAddedIfRoom(newSources));
      } catch (e) {
        dispatch(
          errorActions.errorAdded({
            id: nanoid(),
            error: true,
            message: 'Encountered a problem loading files. Sorry about that.',
          }),
        );
      }
    },
    accept: 'audio/*',
    noClick: true,
    noKeyboard: true,
    multiple: numSourcesAvailable > 1,
    // maxFiles: numSourcesAvailable,
  });

  return (
    <React.Fragment>
      <div className="audioSearchWindow">
        <div {...getRootProps()}>
          <input {...getInputProps()} />
          <Grid container direction="column" spacing={3}>
            <Grid item>
              {!numSources ? (
                <Box pt={3}>
                  <Typography align="center" variant="h4" color="textPrimary">
                    Search for royalty-free music by sonic similarity.
                  </Typography>
                  <Container maxWidth="md">
                    <Typography
                      variant="body1"
                      align="center"
                      color="textSecondary"
                    >
                      {' '}
                      Looking for that perfect song to use in your next video or
                      game?
                      <br />
                      Find something similar to what you already love with an
                      AI-powered search.
                    </Typography>
                  </Container>
                </Box>
              ) : (
                <Box pt={3}>
                  <Container maxWidth="md">
                    <Typography
                      variant="body1"
                      align="justify"
                      color="textSecondary"
                    >
                      Add focus zones to zero in on particular sections and
                      focus on the overall sound, genre, instrumentation, or
                      mood. You can also add multiple tracks and focus on
                      different aspects of each one. None of your music leaves
                      your device; the analysis is done in your browser.
                    </Typography>
                  </Container>
                </Box>
              )}
            </Grid>
            <Box pb={2}>
              <Grid
                container
                item
                justify="center"
                alignItems="center"
                spacing={5}
              >
                <Grid item>
                  <ConditionalWrapper
                    condition={numSourcesAvailable < 1}
                    wrapper={children => (
                      <Tooltip title="Clear a track to add more.">
                        {children}
                      </Tooltip>
                    )}
                  >
                    <span>
                      <Button
                        variant="contained"
                        startIcon={<CloudUploadIcon />}
                        onClick={open}
                        disabled={numSourcesAvailable < 1}
                      >
                        Search by mp3 ({numSourcesAvailable})
                      </Button>
                    </span>
                  </ConditionalWrapper>{' '}
                </Grid>
                <Grid item>{props.SpotifyElement}</Grid>
              </Grid>
            </Box>
            <Grid container item spacing={1}>
              <SearchWindow />
            </Grid>
          </Grid>
        </div>
      </div>
    </React.Fragment>
  );
}

function SearchWindow(props) {
  const [searching, setSearching] = useState(false);
  const [searchProgressLabel, setSearchProgressLabel] = useState<
    string | undefined
  >(undefined);
  const dispatch = useDispatch();
  const { actions: errorActions } = useErrorSlice();
  const { actions: embeddingActions } = useEmbeddingsSlice();
  const numSources = useSelector(selectNumSources);
  const numRegions = useSelector(selectNumRegions);
  const regions = useSelector(selectAllRegions);
  const regionsAvailable = useSelector(selectRegionsAvailable);

  const sources = useSelector(selectAllSources);
  const sourceMap = useSelector(selectAllEntities);
  const classes = useStyles();

  useEffect(() => {
    inferenceWorker.warmupModels();
  }, []);

  const handleSearchClick = useCallback(
    ev => {
      new Promise(async resolve => {
        const embeddings: number[][] = [];
        const showSearchingTimout = setTimeout(() => setSearching(true), 300);
        interface InferenceRegion {
          sourceId: string;
          seconds: number;
          preference: number;
        }
        var hadError = false;
        const inferenceRegions: InferenceRegion[] = [];
        if (numRegions === 0) {
          const samplesPerSource = Math.floor(regionsAvailable / numSources);
          for (const source of sources) {
            if (source.channelData && source.sampleRate) {
              const duration = source.channelData[0].length / source.sampleRate;
              inferenceRegions.push(
                ...[...Array(samplesPerSource).keys()].map(n => ({
                  sourceId: source.id,
                  seconds: ((n + 1) * duration) / (samplesPerSource + 1),
                  preference: 4,
                })),
              );
            }
          }
        } else {
          inferenceRegions.push(
            ...regions.map(r => ({
              sourceId: r.sourceId,
              seconds: r.seconds,
              preference: r.preference,
            })),
          );
        }

        for (let idx = 0; idx < inferenceRegions.length; idx++) {
          const infReg = inferenceRegions[idx];
          setSearchProgressLabel(
            `Resampling zone ${idx + 1} of ${inferenceRegions.length}`,
          );
          try {
            const resampled = await inferenceWorker.downsampleSource(
              sourceMap[infReg.sourceId],
              infReg.seconds,
            );
            setSearchProgressLabel(
              `Analyzing zone ${idx + 1} of ${inferenceRegions.length}`,
            );
            const embedding = await inferenceWorker.runInference(
              resampled,
              infReg.preference,
            );
            embeddings.push(embedding);
          } catch (e) {
            dispatch(
              errorActions.errorAdded({
                id: nanoid(),
                error: true,
                message: 'Error running the machine learning model.',
              }),
            );
            console.log('Error during inference', e);
            hadError = true;
            break;
          }
        }
        clearTimeout(showSearchingTimout);
        setSearching(false);
        setSearchProgressLabel(undefined);
        resolve(embeddings);
        if (!hadError) {
          dispatch(
            embeddingsActions.embeddingsAdded({
              id: nanoid(),
              embeddings,
            }),
          );
        }
      });
    },
    [
      dispatch,
      errorActions,
      numRegions,
      numSources,
      regions,
      regionsAvailable,
      sourceMap,
      sources,
    ],
  );

  return (
    <>
      <Grid
        className={classes.searchForm}
        container
        justify="flex-start"
        direction="column"
        alignItems="stretch"
        spacing={2}
      >
        <Grid item>
          <UploadSurferPair />
        </Grid>
        {numSources > 0 && (
          <Grid item>
            <Box textAlign="center">
              <Button
                color="primary"
                variant="contained"
                size="large"
                disabled={searching}
                autoFocus
                onClick={handleSearchClick}
              >
                {searching ? searchProgressLabel : 'Analyze and Search'}
              </Button>
            </Box>
          </Grid>
        )}
        <Grid item>
          <SearchProgress searching={searching} />
        </Grid>
      </Grid>
    </>
  );
}

export default MusicSearchForm;
