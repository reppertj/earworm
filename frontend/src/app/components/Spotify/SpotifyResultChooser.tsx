import React, {
  useEffect,
  useRef,
  useState,
  useMemo,
  useCallback,
} from 'react';

import IconButton from '@material-ui/core/IconButton';
import Button from '@material-ui/core/Button';
import Dialog from '@material-ui/core/Dialog';
import DialogTitle from '@material-ui/core/DialogTitle';
import DialogActions from '@material-ui/core/DialogActions';
import DialogContent from '@material-ui/core/DialogContent';
import Grid from '@material-ui/core/Grid';
import List from '@material-ui/core/List';
import ListItem from '@material-ui/core/ListItem';
import ListItemText from '@material-ui/core/ListItemText';
import ListItemAction from '@material-ui/core/ListItemSecondaryAction';
import PlayCircleOutlineIcon from '@material-ui/icons/PlayCircleOutline';
import PauseCircleOutlineIcon from '@material-ui/icons/PauseCircleOutline';
import Typography from '@material-ui/core/Typography';
import Tooltip from '@material-ui/core/Tooltip';
import makeStyles from '@material-ui/core/styles/makeStyles';
import Radio from '@material-ui/core/Radio';
import { AudioButton } from '../Playback/AudioButton/AudioButton';
import { useDispatch } from 'react-redux';
import { useAudioSourcesSlice } from '../SearchInputs/AudioInputs/slice';
import { nanoid } from '@reduxjs/toolkit';

const useStyles = makeStyles(theme => ({
  listItem: { cursor: 'pointer' },
}));

function removeUnpreviewableItems(items: any[]): any[] {
  return items.filter(item => {
    return item.preview_url !== null;
  });
}

interface SpotifyResult {
  key: number;
  playing: boolean;
  artist: string | undefined;
  album: string | undefined;
  name: string | undefined;
  previewUrl: string | undefined;
}

function getInfo(item, index: number): SpotifyResult {
  return {
    key: index,
    playing: false,
    artist: item.artists[0]?.name,
    album: item?.album?.name,
    name: item?.name,
    previewUrl: item.preview_url,
  };
}

interface ResultListItemProps {
  item: SpotifyResult;
  selectedValue: number;
  handleChange;
  currentlyPlayingRef: unknown;
  setCurrentlyPlayingRef: React.Dispatch<unknown>;
}

function ResultListItem(props: ResultListItemProps) {
  const classes = useStyles();

  const [isPlaying, setIsPlaying] = useState(false);
  const {
    item,
    selectedValue,
    handleChange,
    currentlyPlayingRef,
    setCurrentlyPlayingRef,
  } = props;
  const primaryText = item.name ? item.name : 'untitled';
  var secondaryText = (item.artist ? item.artist : 'unknown artist') + ' | ';
  secondaryText += item.album ? item.album : 'unknown album';
  const audioElementRef = useRef<HTMLAudioElement | null>(null);

  const selected = useMemo(() => selectedValue === item.key, [
    selectedValue,
    item.key,
  ]);

  const playPauseClickHandler = () => {
    isPlaying ? pauseEvent() : playEvent();
  };

  const playEvent = useCallback(() => {
    if (audioElementRef.current) {
      audioElementRef.current.play();
      setCurrentlyPlayingRef(audioElementRef.current);
      item.playing = true;
      setIsPlaying(true);
    }
  }, [item, setIsPlaying, setCurrentlyPlayingRef]);

  const pauseEvent = useCallback(() => {
    if (audioElementRef.current) {
      audioElementRef.current.pause();
      item.playing = false;
      setIsPlaying(false);
    }
  }, [item, setIsPlaying]);

  const PlayPauseIcon = useMemo(() => {
    return (
      <>{isPlaying ? <PauseCircleOutlineIcon /> : <PlayCircleOutlineIcon />}</>
    );
  }, [isPlaying]);

  const handleClick = () => {
    handleChange({ target: { value: item.key.toString() } });
  };

  useEffect(() => {
    const audioElement = audioElementRef.current;
    if (audioElement) {
      audioElement.addEventListener('play', playEvent);
      audioElement.addEventListener('pause', pauseEvent);
      if (audioElement !== currentlyPlayingRef) {
        pauseEvent();
      }
    }
  }, [pauseEvent, playEvent, currentlyPlayingRef]);

  return (
    <ListItem key={item.key}>
      <Radio
        color="secondary"
        checked={selected}
        onChange={handleChange}
        value={item.key}
        inputProps={{ 'aria-label': item.name }}
      />
      <ListItemText
        className={classes.listItem}
        onClick={handleClick}
        primary={primaryText}
        secondary={secondaryText}
      ></ListItemText>
      <ListItemAction onClick={handleClick}>
        <AudioButton audioElementRef={audioElementRef} url={item.previewUrl} />
        <div
          className="play-pause-button-wrapper"
          onClick={playPauseClickHandler}
        >
          <IconButton edge="end" aria-label="play">
            {PlayPauseIcon}
          </IconButton>
        </div>
      </ListItemAction>
    </ListItem>
  );
}

interface SpotifyChooserProps {
  spotifyResults: any; // TODO: Fix types for this, json response from Spotify API
  setSpotifyResults: React.Dispatch<any>;
  open: true;
  setOpen: React.Dispatch<React.SetStateAction<boolean>>;
}

export function SpotifyChooser(props: SpotifyChooserProps) {
  const [selectedValue, setSelectedValue] = useState<number>(0);
  const [currentlyPlayingRef, setCurrentlyPlayingRef] = useState<unknown>(null);
  const { open, setOpen } = props;
  const dispatch = useDispatch();
  const { actions } = useAudioSourcesSlice();

  const items: Array<any> = props.spotifyResults.tracks.items;
  const processed = removeUnpreviewableItems(items).map(getInfo).slice(0, 30);

  const handleChange = useCallback(
    ev => {
      setSelectedValue(parseInt(ev.target.value));
    },
    [setSelectedValue],
  );

  const handleCancel = () => {
    setOpen(false);
    props.setSpotifyResults(null);
  };
  const handleOk = () => {
    const previewUrl = processed.filter(item => item.key === selectedValue)[0]
      .previewUrl;
    if (previewUrl) {
      dispatch(
        actions.sourcesAddedIfRoom([
          {
            id: nanoid(),
            originalData: previewUrl,
            ready: false,
          },
        ]),
      );
    }
    setOpen(false);
    props.setSpotifyResults(null);
  };

  const ResultListItems = processed.map(item =>
    ResultListItem({
      item,
      handleChange,
      selectedValue,
      currentlyPlayingRef,
      setCurrentlyPlayingRef,
    }),
  );

  return (
    <>
      <Dialog
        disableBackdropClick
        disableEscapeKeyDown
        maxWidth="md"
        open={open}
      >
        <DialogTitle>
          <Grid container justify="space-between">
            <Grid item>Select a song</Grid>
            <Grid item>
              <Tooltip
                disableFocusListener
                interactive
                title={
                  "I can only analyze tracks for which Spotify provides a 30 second preview, but Spotify doesn't provide this preview for all tracks."
                }
              >
                <Typography variant="caption">Something missing?</Typography>
              </Tooltip>
            </Grid>
          </Grid>
        </DialogTitle>
        <DialogContent dividers={true}>
          <List dense={true}>{ResultListItems}</List>
        </DialogContent>
        <DialogActions>
          <Button autoFocus onClick={handleCancel} color="primary">
            Cancel
          </Button>
          <Button onClick={handleOk} color="primary" autoFocus>
            Ok
          </Button>
        </DialogActions>
      </Dialog>
    </>
  );
}
