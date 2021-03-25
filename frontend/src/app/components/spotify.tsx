import React, {
  ReactElement,
  useEffect,
  useRef,
  useState,
  useMemo,
  useCallback,
  useContext,
} from 'react';
import ls from 'local-storage';

import InputAdornment from '@material-ui/core/InputAdornment';
import Snackbar from '@material-ui/core/Snackbar';
import MuiAlert from '@material-ui/lab/Alert';
import AlertTitle from '@material-ui/lab/AlertTitle';
import GREY from '@material-ui/core/colors/grey';
import IconButton from '@material-ui/core/IconButton';
import TextField from '@material-ui/core/TextField';
import CircularProgress from '@material-ui/core/CircularProgress';
import Button from '@material-ui/core/Button';
import Dialog from '@material-ui/core/Dialog';
import DialogTitle from '@material-ui/core/DialogTitle';
import DialogActions from '@material-ui/core/DialogActions';
import DialogContent from '@material-ui/core/DialogContent';
import List from '@material-ui/core/List';
import ListItem from '@material-ui/core/ListItem';
import ListItemIcon from '@material-ui/core/ListItemIcon';
import ListItemText from '@material-ui/core/ListItemText';
import ListItemAction from '@material-ui/core/ListItemSecondaryAction';
import PlayCircleOutlineIcon from '@material-ui/icons/PlayCircleOutline';
import PauseCircleOutlineIcon from '@material-ui/icons/PauseCircleOutline';
import RadioGroup from '@material-ui/core/RadioGroup';
import Radio from '@material-ui/core/Radio';
import { makeStyles, Theme } from '@material-ui/core/styles';
import Grid from '@material-ui/core/Grid';
import SvgIcon, { SvgIconProps } from '@material-ui/core/SvgIcon';
import AudioButton from './AudioButton';
import { ThemeContext } from 'styled-components';

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

const NowPlayingContext = React.createContext<number | null>(null);
const SelectedContext = React.createContext<number | null>(null);

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
}

function ResultListItem(props: ResultListItemProps) {
  const [isPlaying, setIsPlaying] = useState(false);
  const item = props.item;
  const primaryText = item.name ? item.name : 'untitled';
  var secondaryText = (item.artist ? item.artist : 'unknown artist') + ' ⸱ ';
  secondaryText += item.album ? item.album : 'unknown album';
  const audioElementRef = useRef<HTMLAudioElement | null>(null);

  const playPauseClickHandler = () => {
    isPlaying ? pauseEvent() : playEvent();
  };

  const playEvent = useCallback(() => {
    if (audioElementRef.current) {
      audioElementRef.current.play();
    }
    item.playing = true;
    setIsPlaying(true);
    console.log('play clicked');
  }, [audioElementRef.current]);

  const pauseEvent = useCallback(() => {
    if (audioElementRef.current) {
      audioElementRef.current.pause();
    }
    item.playing = false;
    setIsPlaying(false);
    console.log('pause clicked');
  }, [audioElementRef.current]);

  const PlayPauseIcon = useMemo(() => {
    return (
      <>{isPlaying ? <PauseCircleOutlineIcon /> : <PlayCircleOutlineIcon />}</>
    );
  }, [isPlaying]);

  const audioElement = audioElementRef.current;

  useEffect(() => {
    if (audioElement) {
      audioElement.addEventListener('play', playEvent);
      audioElement.addEventListener('pause', pauseEvent);
    }
  }, [audioElement, pauseEvent, playEvent]);

  console.log(item, primaryText, secondaryText);
  return (
    <ListItem key={item.key}>
      <ListItemIcon>
        <Radio />
      </ListItemIcon>
      <ListItemText
        primary={primaryText}
        secondary={secondaryText}
      ></ListItemText>
      <ListItemAction>
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
  addSourceIfSpace: (sourceData: string | File) => void;
}

export function SpotifyChooser(props: SpotifyChooserProps) {
  const { open, setOpen } = props;
  console.log('results', props.spotifyResults);
  const items: Array<any> = props.spotifyResults.tracks.items;
  const processed = removeUnpreviewableItems(items).map(getInfo).slice(0, 20);
  console.log('filtered', processed);

  const handleCancel = () => {
    setOpen(false);
    props.setSpotifyResults(null);
  };
  const handleOk = () => {};

  const ResultListItems = processed.map(item => ResultListItem({ item: item }));
  console.log(processed);

  const value = 0;
  const handleChange = () => {};

  return (
    <>
      <Dialog
        disableBackdropClick
        disableEscapeKeyDown
        maxWidth="lg"
        open={open}
      >
        <DialogTitle>Select a song</DialogTitle>
        <DialogContent dividers={true}>
          <RadioGroup value={value} onChange={handleChange}>
            <List dense={true}>{ResultListItems}</List>
          </RadioGroup>
        </DialogContent>
        <DialogActions>
          <Button autoFocus onClick={handleCancel} color="primary">
            Cancel
          </Button>
          <Button onClick={handleOk} color="primary">
            Ok
          </Button>
        </DialogActions>
      </Dialog>
    </>
  );
}
