import React, { useCallback, useEffect, useMemo, useState } from 'react';
import { useDispatch, useSelector } from 'react-redux';
import {
  selectAllSources,
  selectNumSources,
  selectSourcesAvailable,
} from './slice/selectors';
import { selectRegionsAvailable } from './Regions/slice/selectors';
import { useErrorSlice } from '../Error/slice';

import Box from '@material-ui/core/Box';
import Container from '@material-ui/core/Container';
import makeStyles from '@material-ui/core/styles/makeStyles';
import Grid from '@material-ui/core/Grid';
import Typography from '@material-ui/core/Typography';
import Button from '@material-ui/core/Button';
import CloudUploadIcon from '@material-ui/icons/CloudUpload';
import Tooltip from '@material-ui/core/Tooltip';

import { useDropzone } from 'react-dropzone';
import getChannelDataAndSampleRate from '../../utils/audio';
import { ConditionalWrapper } from '../ConditionalWrapper';
import { UploadSurferPair } from '../UploadSurfer';
import { ErrorSnackBar } from '../Error/ErrorSnackBar';
import SpotifySearchForm from '../Spotify/SpotifySearchForm';
import SearchProgress from '../SearchProgress';
import { SpotifyAuthorizationButton } from '../Spotify/SpotifyAuthorizationButton';
import { SpotifyChooser } from '../Spotify/SpotifyResultChooser';
import Worker from '../../worker';
import { selectSpotifyAuth } from '../Spotify/slice/selectors';
import { useAudioSourcesSlice } from './slice';
import { Source } from './slice/types';
import { nanoid } from '@reduxjs/toolkit';

var inferenceWorker = new Worker();

const useStyles = makeStyles(theme => ({
  searchForm: {
    padding: theme.spacing(2),
  },
}));

// interface RegionStart {
//   id: number;
//   seconds: number;
//   preference: number;
// }

// interface Source {
//   id: number;
//   regionStarts: RegionStart[];
//   source?: string | File;
//   channelData?: FullAudioData;
// }

// type Sources = Array<Source | null>;

// async function getMaybeMultipleChannelData(sources: Sources): Promise<Sources> {
//   const newSources = sources.slice();
//   for (let i = 0; i < sources.length; i++) {
//     const source = newSources[i];
//     if (source && source.source) {
//       const channelData = await getChannelDataAndSampleRate(source.source);
//       newSources[i] = { channelData, ...source };
//     }
//   }
//   return newSources;
// }

function MusicSearchForm(props) {
  const [hasError, setHasError] = useState({ error: false, message: '' });
  const [spotifyResults, setSpotifyResults] = useState<any>(null);
  const [chooserOpen, setChooserOpen] = useState(false);

  const { loggedIn, tokenExpiryDate } = useSelector(selectSpotifyAuth);

  const sources = useSelector(selectAllSources);

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
    if (!chooserOpen && spotifyResults) {
      if (!spotifyResults.tracks) {
        setHasError({
          error: true,
          message:
            "Couldn't process the results from Spotify. Sorry about that.",
        });
      } else if (spotifyResults.tracks.items.length < 1) {
        setHasError({
          error: true,
          message: 'No results! Try searching for something else.',
        });
      } else {
        setChooserOpen(true);
      }
    }
  }, [spotifyResults, chooserOpen]);

  return (
    <>
      <UploadWindow SpotifyElement={SpotifyElement} />
      {hasError.error && (
        <ErrorSnackBar hasError={hasError} setHasError={setHasError} />
      )}
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
        errorActions.errorAdded({
          error: true,
          message: 'Encountered a problem loading files. Sorry about that.',
        });
        console.log(e);
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
              {numSourcesAvailable > 1 && (
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
                      Looking for that perfect song to use in your next video or
                      game?
                      <br />
                      Find something similar to what you already love with an
                      AI-powered search.
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
  const [results, setResults] = useState<Float32Array[]>(); // Placeholder while figuring out API
  const numSources = useSelector(selectNumSources);

  const sources = useSelector(selectAllSources);
  const classes = useStyles();

  useEffect(() => {
    inferenceWorker.warmupModels();
  }, []);

  const handleSearchClick = useCallback(
    ev => {
      // getMaybeMultipleChannelData(sources).then(sources => {
      //   let sourcesNoFile: Sources = [];
      //   for (let source of sources) {
      //     if (source && source.channelData) {
      //       sourcesNoFile.push({
      //         id: source.id,
      //         regionStarts: source.regionStarts,
      //         channelData: source.channelData,
      //       });
      //     }
      //   }
      //   const processed: Promise<Float32Array[] | undefined> = new Promise(
      //     async resolve => {
      //       const embeddings: Float32Array[] = [];
      //       const showSearchingTimout = setTimeout(
      //         () => setSearching(true),
      //         300,
      //       );
      //       for (let source of sourcesNoFile) {
      //         if (source !== null) {
      //           if (source.regionStarts.length > 0) {
      //             for (let idx = 0; idx < source.regionStarts.length; idx++) {
      //               setSearchProgressLabel(`Resampling`);
      //               const resampledData = await inferenceWorker.downsampleSource(
      //                 source,
      //                 idx,
      //               );
      //               setSearchProgressLabel('Analyzing spectrum');
      //               console.log('time', source.regionStarts[idx].seconds);
      //               const embedding = await inferenceWorker.runInference(
      //                 resampledData,
      //                 source.regionStarts[idx].preference,
      //               );
      //               embeddings.push(embedding);
      //             }
      //           } else {
      //             setSearchProgressLabel(`Resampling`);
      //             const resampledData = await inferenceWorker.downsampleSource(
      //               source,
      //               0,
      //             );
      //             setSearchProgressLabel('Analyzing spectrum');
      //             const embedding = await inferenceWorker.runInference(
      //               resampledData,
      //               4,
      //             );
      //             embeddings.push(embedding);
      //           }
      //         }
      //       }
      //       if (sourcesNoFile[0] !== null) {
      //         console.log(embeddings);
      //         setResults(embeddings);
      //         resolve(embeddings);
      //         clearTimeout(showSearchingTimout);
      //         setSearching(false);
      //         setSearchProgressLabel(undefined);
      //       }
      //     },
      //   );
      // });
    },
    [sources],
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
                onClick={handleSearchClick}
              >
                Analyze and Search
              </Button>
            </Box>
          </Grid>
        )}
        <Grid item>
          <SearchProgress searching={searching} label={searchProgressLabel} />
        </Grid>
        {results}
      </Grid>
    </>
  );
}

export default MusicSearchForm;
