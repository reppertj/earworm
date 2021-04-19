import React, { memo, useEffect, useRef, useState } from 'react';

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

import { useSelector, useDispatch } from 'react-redux';
import { selectVolume } from '../../Playback/volumeSlice/selectors';

import WaveSurfer from 'wavesurfer.js';
import RegionsPlugin from 'wavesurfer.js/dist/plugin/wavesurfer.regions';
import RegionPreferences from '../Regions/RegionPreferences';

import {
  selectRegionsAvailable,
  selectAllRegions,
  selectRegionIdToAlphaColorMap,
  selectRegionIdToSolidColorMap,
  selectRegionIdToShowIdMap,
} from '../Regions/slice/selectors';
import { useRegionsSlice } from '../Regions/slice';
import { nanoid } from '@reduxjs/toolkit';
import { useAudioSourcesSlice } from './slice';
import { selectAllSources, selectSourceByID } from './slice/selectors';

const useStyles = makeStyles(theme =>
  createStyles({
    button: {
      margin: theme.spacing(1),
    },
  }),
);

// interface UploadSurferPairProps {}

function UploadSurferGroup() {
  const sources = useSelector(selectAllSources);
  const regions = useSelector(selectAllRegions);

  return (
    <>
      <Grid container direction="column" spacing={2}>
        <Grid container item direction="column" spacing={2}>
          {sources.map(source => {
            return (
              <div key={source.id}>
                <Box padding={2}>
                  <Grid item>
                    <UploadSurfer sourceId={source.id} />
                  </Grid>
                </Box>
              </div>
            );
          })}
        </Grid>
        <Grid container justify="space-around" item direction="row" spacing={1}>
          {regions.map(region => {
            return (
              <div key={region.id}>
                <Grid item>
                  <Paper>
                    <RegionPreferences regionId={region.id} />
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
  sourceId: string;
}

const UploadSurfer = memo((props: UploadSurferProps) => {
  const { sourceId } = props;

  const classes = useStyles();
  const theme = useTheme();

  const makeSurferOptions = (ref: HTMLDivElement) => ({
    container: ref,
    waveColor: theme.palette.primary.light,
    progressColor: theme.palette.secondary.light,
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

  const waveformRef = useRef<HTMLDivElement | null>(null);
  const wavesurfer = useRef<WaveSurfer | null>(null);
  const [loaded, setLoaded] = useState(false);
  const [playing, setPlaying] = useState<boolean>(false);

  const volume = useSelector(selectVolume).value;
  const source = useSelector(state => selectSourceByID(state, sourceId));
  const regions = useSelector(selectAllRegions).filter(
    region => region.sourceId === sourceId,
  );
  const numRegionsAvailable = useSelector(selectRegionsAvailable);

  const dispatch = useDispatch();
  const { actions: regionActions } = useRegionsSlice();
  const { actions: sourceActions } = useAudioSourcesSlice();

  // Create new instance on mount and when source changes
  useEffect(() => {
    if (waveformRef.current) {
      // onPlayPause(false, sourceId);
      const options = makeSurferOptions(waveformRef.current);
      wavesurfer.current = WaveSurfer.create(options);
      const audioData = source?.originalData;
      if (audioData) {
        if (typeof audioData === 'string') {
          wavesurfer.current.load(audioData);
        } else {
          wavesurfer.current.loadBlob(audioData);
        }
        setLoaded(true);
      }
    }
    return () => {
      if (wavesurfer.current) {
        wavesurfer.current?.destroy();
        setLoaded(false);
      }
    };
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [source, sourceId]);

  // useEffect(() => {
  //   playing ? wavesurfer.current?.play() : wavesurfer.current?.pause();
  // }, [playing]);

  // Initial Volume
  wavesurfer.current?.on('ready', function () {
    if (wavesurfer.current) {
      setLoaded(true);
      wavesurfer.current.setVolume(volume);
    }
  });

  wavesurfer.current?.on('finish', function () {
    if (wavesurfer.current) {
      setPlaying(false);
      wavesurfer.current.stop();
    }
  });

  // Later Volume
  useEffect(() => {
    wavesurfer.current?.setVolume(volume);
  }, [volume]);

  // Update region attributes inside wavesurfer

  const alphaColorMap = useSelector(selectRegionIdToAlphaColorMap);
  const solidColorMap = useSelector(selectRegionIdToSolidColorMap);
  const showIdMap = useSelector(selectRegionIdToShowIdMap);

  if (wavesurfer.current && loaded) {
    const surferRegions = Object.values(wavesurfer.current.regions.list);
    surferRegions.forEach((surferRegion: any) => {
      surferRegion.update({
        color: alphaColorMap[surferRegion.attributes.regionId],
        attributes: {
          regionId: surferRegion.attributes.regionId,
          label: showIdMap[surferRegion.attributes.regionId],
        },
      });
    });
  }

  function addRegion() {
    if (wavesurfer.current && loaded && sourceId) {
      const start = Math.min(
        wavesurfer.current.getCurrentTime(),
        wavesurfer.current.getDuration() - 3,
      );
      const regionId = nanoid();
      dispatch(
        regionActions.regionAddedIfRoom({
          id: regionId,
          sourceId: sourceId,
          seconds: start,
          preference: 4,
        }),
      );
      const surferRegion = wavesurfer.current.addRegion({
        start,
        end: start + 3,
        loop: false,
        drag: true,
        resize: false,
        color: alphaColorMap[regionId],
        attributes: { regionId, label: '' },
      });
      surferRegion.regionId = regionId;
      surferRegion.on('update-end', function () {
        dispatch(
          regionActions.regionUpdated({
            id: regionId,
            changes: { seconds: surferRegion.start as number },
          }),
        );
      });
    }
  }

  function removeRegion(regionId: string) {
    if (wavesurfer.current && loaded) {
      const waveSurferRegionObject: any = Object.values(
        wavesurfer.current.regions.list,
      ).find((elem: any) => elem.attributes.regionId === regionId);
      if (waveSurferRegionObject) {
        waveSurferRegionObject.remove();
      }
      dispatch(regionActions.regionRemoved(regionId));
    }
  }

  function removeSource() {
    if (wavesurfer.current) {
      // hack b/c wavesurfer has a memory leak and doesn't free all references on destory
      wavesurfer.current.load('%PUBLIC_URL%/250-milliseconds-of-silence.mp3');
      wavesurfer.current.destroy();
      wavesurfer.current = null;
      setLoaded(false);
    }
    if (source?.originalData) {
      URL.revokeObjectURL(source.originalData);
    }
    dispatch(regionActions.regionsRemoved(regions.map(region => region.id)));
    dispatch(sourceActions.sourceRemoved(sourceId));
  }

  const handlePlayPause = () => {
    if (wavesurfer.current) {
      wavesurfer.current.playPause();
      setPlaying(wavesurfer.current.isPlaying());
      // onPlayPause(nowPlaying, sourceId);
    }
  };

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
                  startIcon={loaded && !playing ? <PlayIcon /> : <PauseIcon />}
                  className={classes.button}
                  onClick={handlePlayPause}
                  disabled={!loaded}
                >
                  {!playing ? 'Play' : 'Pause'}
                </Button>
                <Tooltip
                  title={
                    'Focus zones identify exactly what sounds the search should try to match. You can add up to six.'
                  }
                >
                  <span>
                    <Button
                      onClick={addRegion}
                      variant="outlined"
                      className={classes.button}
                      size="small"
                      disabled={!loaded || !numRegionsAvailable}
                    >
                      Add Focus Zone ({numRegionsAvailable})
                    </Button>
                  </span>
                </Tooltip>
                {regions &&
                  regions.map(region => {
                    if (region) {
                      return (
                        <Button
                          size="small"
                          className={classes.button}
                          // variant="outlined"
                          color="primary"
                          key={region.id}
                          style={{
                            color: solidColorMap[region.id],
                            // border: `1px solid ${solidColorMap[region.id]}`,
                          }}
                          onClick={ev => removeRegion(region.id)}
                        >
                          Remove Zone {showIdMap[region.id]}
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
});

export { UploadSurferGroup as UploadSurferPair };
