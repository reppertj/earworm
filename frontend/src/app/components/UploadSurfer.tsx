import React, { useEffect, useRef, useState } from 'react';

import Box from '@material-ui/core/Box';
import Grid from '@material-ui/core/Grid';
import Button from '@material-ui/core/Button';
import PlayIcon from '@material-ui/icons/PlayArrow';
import Paper from '@material-ui/core/Paper';
import PauseIcon from '@material-ui/icons/Pause';
import Tooltip from '@material-ui/core/Tooltip';
import makeStyles from '@material-ui/core/styles/makeStyles';
import createStyles from '@material-ui/core/styles/createStyles';
import useTheme from '@material-ui/core/styles/useTheme';
import DeleteIcon from '@material-ui/icons/DeleteOutline';

import WaveSurfer from 'wavesurfer.js';
import RegionsPlugin from 'wavesurfer.js/dist/plugin/wavesurfer.regions';
import RegionPreferences from './RegionPreferences';
import { Source } from './SearchForm';

const useStyles = makeStyles(theme =>
  createStyles({
    button: {
      margin: theme.spacing(1),
    },
  }),
);

const regionColors = [
  'rgba(120, 144, 156, 0.5)', // Primary w/ alpha
  'rgba(255, 202, 40, 0.5)', // Secondary w/ alpha
];

export const regionColorsMui: Array<'primary' | 'secondary'> = [
  'primary',
  'secondary',
];

interface RegionStartCallbacks {
  addRegionStart: (sourceID: any, startTime: any) => number | undefined;
  removeRegionStart: (sourceID: any, regionID: any) => void;
  updateRegionStart: (sourceID: any, regionID: any, startTime: any) => void;
  updateRegionPreference: (regionID: any, newPreference: any) => void;
}

interface UploadSurferPairProps {
  removeSource: (sourcePosition: number) => void;
  sources: Source[];
  numAvailableRegions: number;
  regionStartCallbacks: RegionStartCallbacks;
}

function UploadSurferPair(props: UploadSurferPairProps) {
  const [playing, setPlaying] = useState([false, false]);
  const { sources, numAvailableRegions, removeSource } = props;
  const {
    addRegionStart,
    removeRegionStart,
    updateRegionStart,
    updateRegionPreference,
  } = props.regionStartCallbacks;

  const onPlayPause = (isPlaying: boolean, index: number) => {
    const other = index === 0 ? 1 : 0;
    const nowPlaying = playing.slice();
    nowPlaying[index] = isPlaying;
    if (isPlaying) {
      nowPlaying[other] = !isPlaying;
    }
    setPlaying(nowPlaying);
  };

  function makeRegionOptions(id: number, start: number) {
    console.log();
    return {
      id: id.toString(),
      start: start,
      end: start + 3,
      loop: true,
      drag: true,
      resize: false,
      color: regionColors[id - 1],
    };
  }

  const handleAddRegion = (sourceID: number, waveSurfer: WaveSurfer) => {
    if (numAvailableRegions) {
      const startPosition = Math.min(
        waveSurfer.getCurrentTime(),
        waveSurfer.getDuration() - 3,
      );
      const regionID = addRegionStart(sourceID, startPosition);
      if (regionID !== undefined) {
        const region = waveSurfer.addRegion(
          makeRegionOptions(regionID, startPosition),
        );
        region.regionID = regionID;
        region.on('update-end', function () {
          updateRegionStart(sourceID, regionID, region.start);
        });
      }
    }
  };

  const regionStarts = sources.flatMap(source => source?.regionStarts || []);

  return (
    <>
      <Grid container direction="column" spacing={2}>
        <Grid container item direction="column" spacing={2}>
          {sources.map((elem, idx) => {
            return (
              <div key={idx}>
                {elem && (
                  <Box padding={2}>
                    <Grid item>
                      <UploadSurfer
                        surferID={idx}
                        source={elem}
                        removeSource={() => removeSource(idx)}
                        handleAddRegion={handleAddRegion}
                        handleRemoveRegion={regionID =>
                          removeRegionStart(elem.id, regionID)
                        }
                        numAvailableRegions={numAvailableRegions}
                        playing={playing[idx]}
                        onPlayPause={(isPlaying: boolean) =>
                          onPlayPause(isPlaying, idx)
                        }
                      />
                    </Grid>
                  </Box>
                )}
              </div>
            );
          })}
        </Grid>
        <Grid container justify="space-around" item direction="row" spacing={1}>
          {regionStarts.map(regionStart => {
            return (
              <div key={regionStart.id}>
                <Grid item>
                  <Paper>
                    <RegionPreferences
                      regionID={regionStart.id}
                      regionPreference={regionStart.preference}
                      updateRegionPreference={updateRegionPreference}
                    />
                  </Paper>
                </Grid>
              </div>
            );
          })}
        </Grid>
      </Grid>
    </>
  );
}

interface UploadSurferProps {
  surferID: number;
  source: Source;
  onPlayPause: (isPlaying: boolean, index: number) => void;
  playing: boolean;
  removeSource: () => void;
  handleAddRegion: (sourceID: number, waveSurfer: WaveSurfer) => void;
  handleRemoveRegion: (regionID: number) => void;
  numAvailableRegions: number;
}

function UploadSurfer(props: UploadSurferProps) {
  const waveformRef = useRef<HTMLDivElement | null>(null);
  const wavesurfer = useRef<WaveSurfer | null>(null);
  const [volume, setVolume] = useState(0.8);
  const [loaded, setLoaded] = useState(false);

  const {
    source,
    surferID,
    removeSource,
    handleAddRegion,
    handleRemoveRegion,
    numAvailableRegions,
    onPlayPause,
    playing,
  } = props;

  const sourcePath = source?.source === undefined ? null : source.source;

  const classes = useStyles();
  const theme = useTheme();

  const makeSurferOptions = (ref: HTMLDivElement) => ({
    container: ref,
    waveColor: theme.palette.primary.light,
    progressColor: theme.palette.secondary.main,
    cursorColor: theme.palette.secondary.dark,
    barWidth: 2,
    barHeight: 1,
    responsive: true,
    interact: true,
    scrollParent: true,
    height: 80,
    normalize: true,
    partialRender: true,
    plugins: [RegionsPlugin.create()],
  });

  // Create new instance on mount and when source changes
  useEffect(() => {
    if (waveformRef.current) {
      onPlayPause(false, surferID);
      const options = makeSurferOptions(waveformRef.current);
      wavesurfer.current = WaveSurfer.create(options);
      if (sourcePath !== null) {
        if (typeof sourcePath === 'string') {
          wavesurfer.current.load(sourcePath);
        } else {
          wavesurfer.current.loadBlob(sourcePath);
        }
      }
    }
    return () => {
      if (wavesurfer.current) {
        wavesurfer.current?.destroy();
        setLoaded(false);
      }
    };
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [sourcePath, surferID]);

  useEffect(() => {
    playing ? wavesurfer.current?.play() : wavesurfer.current?.pause();
  }, [playing]);

  // Set volume in case of new instance with volume already set
  wavesurfer.current?.on('ready', function () {
    if (wavesurfer.current) {
      setLoaded(true);
      wavesurfer.current.setVolume(volume);
      setVolume(volume);
    }
  });

  function addRegion() {
    if (wavesurfer.current && loaded) {
      if (numAvailableRegions) {
        handleAddRegion(props.source.id, wavesurfer.current);
      }
    }
    console.log(wavesurfer.current);
  }

  function removeRegion(regionID: number) {
    if (wavesurfer.current && loaded) {
      const waveSurferRegionObject: any = Object.values(
        wavesurfer.current.regions.list,
      ).find((elem: any) => elem.regionID === regionID);
      waveSurferRegionObject.remove();
      handleRemoveRegion(regionID);
    }
  }

  const handlePlayPause = () => {
    if (wavesurfer.current) {
      wavesurfer.current.playPause();
      const nowPlaying = wavesurfer.current.isPlaying();
      onPlayPause(nowPlaying, surferID);
    }
  };

  function onVolumeChange(ev: any, newVolume: number | number[]) {
    setVolume(newVolume as number);
    wavesurfer.current?.setVolume(newVolume as number);
  }

  return (
    <>
      <Paper elevation={3}>
        <div>
          <Box padding={1}>
            <div id="waveform" ref={waveformRef} />
          </Box>
          <div className="controls">
            <Grid container justify="space-between" alignItems="flex-start">
              <Grid item xs={5}>
                <Button
                  variant="contained"
                  size="small"
                  startIcon={!props.playing ? <PlayIcon /> : <PauseIcon />}
                  className={classes.button}
                  onClick={handlePlayPause}
                  disabled={!loaded}
                >
                  {!props.playing ? 'Play' : 'Pause'}
                </Button>
                <Tooltip
                  title={
                    'Focus zones identify exactly what sounds the search should try to match. You can add up to two.'
                  }
                >
                  <span>
                    <Button
                      onClick={addRegion}
                      variant="outlined"
                      className={classes.button}
                      size="small"
                      disabled={!loaded || !numAvailableRegions}
                    >
                      Add Focus Zone ({numAvailableRegions})
                    </Button>
                  </span>
                </Tooltip>
                {props.source &&
                  props.source.regionStarts.map(region => {
                    if (region) {
                      return (
                        <Button
                          size="small"
                          className={classes.button}
                          variant="outlined"
                          color={regionColorsMui[region.id - 1]}
                          key={region.id}
                          onClick={ev => removeRegion(region.id)}
                        >
                          Remove Zone {region.id}
                        </Button>
                      );
                    } else {
                      return null;
                    }
                  })}
              </Grid>
              <Grid container item xs={5} alignItems="center">
                {loaded && (
                  <Grid item>
                    <Button
                      variant="outlined"
                      size="small"
                      color="secondary"
                      startIcon={<DeleteIcon />}
                      className={classes.button}
                      disabled={!loaded}
                      onClick={removeSource}
                    >
                      Clear track
                    </Button>
                  </Grid>
                )}
              </Grid>
            </Grid>
          </div>
        </div>
      </Paper>
    </>
  );
}

export { UploadSurferPair };
