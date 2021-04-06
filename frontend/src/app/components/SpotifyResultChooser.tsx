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
import List from '@material-ui/core/List';
import ListItem from '@material-ui/core/ListItem';
import ListItemText from '@material-ui/core/ListItemText';
import ListItemAction from '@material-ui/core/ListItemSecondaryAction';
import PlayCircleOutlineIcon from '@material-ui/icons/PlayCircleOutline';
import PauseCircleOutlineIcon from '@material-ui/icons/PauseCircleOutline';
import Radio from '@material-ui/core/Radio';
import AudioButton from './AudioButton';

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
  handleChange:
    | ((event: React.ChangeEvent<HTMLInputElement>, checked: boolean) => void)
    | undefined;
}

function ResultListItem(props: ResultListItemProps) {
  const [isPlaying, setIsPlaying] = useState(false);
  const { item, selectedValue, handleChange } = props;
  const primaryText = item.name ? item.name : 'untitled';
  var secondaryText = (item.artist ? item.artist : 'unknown artist') + ' ⸱ ';
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
    }
    item.playing = true;
    setIsPlaying(true);
    console.log('play clicked');
  }, [item, setIsPlaying]);

  const pauseEvent = useCallback(() => {
    if (audioElementRef.current) {
      audioElementRef.current.pause();
    }
    item.playing = false;
    setIsPlaying(false);
    console.log('pause clicked');
  }, [item, setIsPlaying]);

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

  return (
    <ListItem key={item.key}>
      {/* <ListItemIcon> */}
      <Radio
        checked={selected}
        onChange={handleChange}
        value={item.key}
        inputProps={{ 'aria-label': item.name }}
      />
      {/* </ListItemIcon> */}
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
  const [selectedValue, setSelectedValue] = useState<number>(0);
  const { open, setOpen, addSourceIfSpace } = props;

  const items: Array<any> = props.spotifyResults.tracks.items;
  const processed = removeUnpreviewableItems(items).map(getInfo).slice(0, 20);

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
      addSourceIfSpace(previewUrl);
    }
    setOpen(false);
    props.setSpotifyResults(null);
  };

  const ResultListItems = processed.map(item =>
    ResultListItem({
      item: item,
      handleChange: handleChange,
      selectedValue: selectedValue,
    }),
  );
  console.log(processed);

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
          <List dense={true}>{ResultListItems}</List>
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
