import React, { useCallback, useEffect, useMemo, useState } from 'react';

import Box from '@material-ui/core/Box';
import Container from '@material-ui/core/Container';
import makeStyles from '@material-ui/core/styles/makeStyles';
import Grid from '@material-ui/core/Grid';
import Typography from '@material-ui/core/Typography';
import Button from '@material-ui/core/Button';
import CloudUploadIcon from '@material-ui/icons/CloudUpload';

import { useDropzone } from 'react-dropzone';
import getChannelDataAndSampleRate, { FullAudioData } from '../utils/audio';
import { UploadSurferPair } from './UploadSurfer';
import { ErrorSnackBar } from './ErrorSnackBar';
import SpotifySearchForm from './SpotifySearchForm';
import SearchProgress from './SearchProgress';
import { SpotifyAuthorizationButton } from './Spotify/SpotifyAuthorizationButton';
import { SpotifyChooser } from './SpotifyResultChooser';
import Worker from '../worker';
import { selectSpotifyAuth } from './Spotify/slice/selectors';
import { useSelector } from 'react-redux';

var inferenceWorker = new Worker();

const useStyles = makeStyles(theme => ({
  searchForm: {
    padding: theme.spacing(2),
  },
}));

interface RegionStart {
  id: number;
  seconds: number;
  preference: number;
}

interface Source {
  id: number;
  regionStarts: RegionStart[];
  source?: string | File;
  channelData?: FullAudioData;
}

type Sources = Array<Source | null>;

async function getMaybeMultipleChannelData(sources: Sources): Promise<Sources> {
  const newSources = sources.slice();
  for (let i = 0; i < sources.length; i++) {
    const source = newSources[i];
    if (source && source.source) {
      const channelData = await getChannelDataAndSampleRate(source.source);
      newSources[i] = { channelData, ...source };
    }
  }
  return newSources;
}

function MusicSearchForm(props) {
  // const [loggedIn, setLoggedIn] = useState(false);
  // const [accessToken, setAccessToken] = useState<string | null>(null);
  // const [tokenExpiryDate, setTokenExpiryDate] = useState<number | null>(null);
  const [hasError, setHasError] = useState({ error: false, message: '' });
  const [spotifyResults, setSpotifyResults] = useState<any>(null);
  const [chooserOpen, setChooserOpen] = useState(false);
  const [sources, setSources] = useState<Sources>([null, null]);
  const [filesLeft, setFilesLeft] = useState(2);

  const { loggedIn, tokenExpiryDate } = useSelector(selectSpotifyAuth);

  const addSources = useCallback(
    (sourceData: Array<string | File>) => {
      let newSources = sources.slice();
      for (let data of sourceData) {
        let openPosition = newSources.findIndex(elem => elem === null);
        if (openPosition === -1) {
          openPosition = newSources.length - 1;
        }
        console.log(openPosition);
        newSources[openPosition] = {
          id: openPosition,
          source: data,
          regionStarts: [],
        };
      }
      console.log(newSources);
      setSources(newSources);
      setFilesLeft(newSources.filter(newSources => newSources === null).length);
    },
    [sources],
  );

  const removeSource = useCallback(
    (sourcePosition: number) => {
      const newSources = sources.slice();
      if ([0, 1].includes(sourcePosition)) {
        newSources[sourcePosition] = null;
        setSources(newSources);
        setFilesLeft(newSources.filter(source => source === null).length);
      }
    },
    [sources],
  );

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
      <UploadWindow
        sources={sources}
        setSources={setSources}
        removeSource={removeSource}
        addSources={addSources}
        filesLeft={filesLeft}
        setFilesLeft={setFilesLeft}
        SpotifyElement={SpotifyElement}
      />
      {hasError.error && (
        <ErrorSnackBar hasError={hasError} setHasError={setHasError} />
      )}
      {chooserOpen && (
        <SpotifyChooser
          spotifyResults={spotifyResults}
          setSpotifyResults={setSpotifyResults}
          open={chooserOpen}
          setOpen={setChooserOpen}
          addSources={addSources}
        />
      )}
    </>
  );
}

function getFreeRegionIDs(sources: Sources) {
  const regionIDs = new Set([1, 2]);
  const usedRegionIDs = sources.flatMap(source => {
    if (source) {
      return source.regionStarts.map(regionStart => regionStart.id);
    }
    return [];
  });
  for (let id of usedRegionIDs) {
    regionIDs.delete(id);
  }
  return Array.from(regionIDs);
}

interface UploadWindowProps {
  sources: Sources;
  setSources: React.Dispatch<React.SetStateAction<Sources>>;
  filesLeft: number;
  setFilesLeft: React.Dispatch<React.SetStateAction<number>>;
  addSources: (sourceData: Array<string | File>) => void;
  removeSource: (sourcePosition: number) => void;
  SpotifyElement: JSX.Element;
}

function UploadWindow(props: UploadWindowProps) {
  const { sources, setSources, filesLeft, addSources, removeSource } = props;

  const { open, getRootProps, getInputProps } = useDropzone({
    onDrop: acceptedFiles => {
      addSources(acceptedFiles);
    },
    accept: 'audio/*',
    noClick: true,
    noKeyboard: true,
    multiple: true,
    maxFiles: 2,
  });

  const numAvailableRegions = getFreeRegionIDs(sources).length;

  const addRegionStart = useCallback(
    (sourceID, startTime) => {
      const regionID = getFreeRegionIDs(sources)[0];
      if (regionID) {
        const newSources = sources.slice();
        const sourceIdx = newSources.findIndex(
          source => source?.id === sourceID,
        );
        newSources[sourceIdx]?.regionStarts.push({
          id: regionID,
          seconds: startTime,
          preference: 4,
        });
        setSources(newSources);
        return regionID;
      }
    },
    [sources, setSources],
  );

  const removeRegionStart = useCallback(
    (sourceID, regionID) => {
      const newSources = sources.slice();
      const sourceIDX = newSources.findIndex(source => source?.id === sourceID);
      const regionIDX = newSources[sourceIDX]?.regionStarts.findIndex(
        start => start.id === regionID,
      );
      if (regionIDX !== undefined && regionIDX !== -1) {
        newSources[sourceIDX]?.regionStarts.splice(regionIDX, 1);
      }
      setSources(newSources);
    },
    [sources, setSources],
  );

  const updateRegionStart = useCallback(
    (sourceID, regionID, startTime) => {
      const newSources = sources.slice();
      const sourceIdx = newSources.findIndex(source => source?.id === sourceID);
      const regionIdx = newSources[sourceIdx]?.regionStarts.findIndex(
        start => start.id === regionID,
      );
      if (regionIdx !== undefined) {
        (newSources[sourceIdx] as Source).regionStarts[
          regionIdx
        ].seconds = startTime;
      }
      setSources(newSources);
    },
    [sources, setSources],
  );

  const updateRegionPreference = useCallback(
    (regionID, newPreference) => {
      const newSources = sources.slice();
      const sourceIdx = newSources.findIndex(source =>
        source?.regionStarts?.map(region => region.id).includes(regionID),
      );
      const regionIdx = newSources[sourceIdx]?.regionStarts.findIndex(
        region => region.id === regionID,
      );
      if (regionIdx !== undefined) {
        (newSources[sourceIdx] as Source).regionStarts[
          regionIdx
        ].preference = newPreference;
      }
      setSources(newSources);
    },
    [sources, setSources],
  );

  return (
    <React.Fragment>
      <div className="audioSearchWindow">
        <div {...getRootProps()}>
          <input {...getInputProps()} />
          <Grid container direction="column" spacing={3}>
            <Grid item>
              {filesLeft > 1 && (
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
                  <Button
                    variant="contained"
                    startIcon={<CloudUploadIcon />}
                    onClick={open}
                    disabled={filesLeft < 1}
                  >
                    Search by mp3 ({filesLeft})
                  </Button>
                </Grid>
                <Grid item>{props.SpotifyElement}</Grid>
              </Grid>
            </Box>
            <Grid container item spacing={1}>
              <SearchWindow
                sources={sources}
                setSources={setSources}
                removeSource={removeSource}
                numAvailableRegions={numAvailableRegions}
                regionStartCallbacks={{
                  removeRegionStart,
                  addRegionStart,
                  updateRegionStart,
                  updateRegionPreference,
                }}
              />
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

  const sources: Sources = props.sources;
  const classes = useStyles();

  const numSources = sources.filter(source => source !== null).length;

  useEffect(() => {
    inferenceWorker.warmupModels();
  });

  const handleSearchClick = useCallback(
    ev => {
      getMaybeMultipleChannelData(sources).then(sources => {
        let sourcesNoFile: Sources = [];
        for (let source of sources) {
          if (source && source.channelData) {
            sourcesNoFile.push({
              id: source.id,
              regionStarts: source.regionStarts,
              channelData: source.channelData,
            });
          }
        }
        const processed: Promise<Float32Array[] | undefined> = new Promise(
          async resolve => {
            const embeddings: Float32Array[] = [];
            const showSearchingTimout = setTimeout(
              () => setSearching(true),
              300,
            );
            for (let source of sourcesNoFile) {
              if (source !== null) {
                if (source.regionStarts.length > 0) {
                  for (let idx = 0; idx < source.regionStarts.length; idx++) {
                    setSearchProgressLabel(`Resampling`);
                    const resampledData = await inferenceWorker.downsampleSource(
                      source,
                      idx,
                    );
                    setSearchProgressLabel('Analyzing spectrum');
                    console.log('time', source.regionStarts[idx].seconds);
                    const embedding = await inferenceWorker.runInference(
                      resampledData,
                      source.regionStarts[idx].preference,
                    );
                    embeddings.push(embedding);
                  }
                } else {
                  setSearchProgressLabel(`Resampling`);
                  const resampledData = await inferenceWorker.downsampleSource(
                    source,
                    0,
                  );
                  setSearchProgressLabel('Analyzing spectrum');
                  const embedding = await inferenceWorker.runInference(
                    resampledData,
                    4,
                  );
                  embeddings.push(embedding);
                }
              }
            }
            if (sourcesNoFile[0] !== null) {
              console.log(embeddings);
              setResults(embeddings);
              resolve(embeddings);
              clearTimeout(showSearchingTimout);
              setSearching(false);
              setSearchProgressLabel(undefined);
            }
          },
        );
      });
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
          <UploadSurferPair
            sources={props.sources}
            removeSource={props.removeSource}
            numAvailableRegions={props.numAvailableRegions}
            regionStartCallbacks={props.regionStartCallbacks}
          />
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
export type { Source, RegionStart };
