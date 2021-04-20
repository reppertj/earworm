/**
 *
 * ResultPlaybackCell
 *
 */
import React, {
  memo,
  useState,
  useRef,
  useMemo,
  useCallback,
  useEffect,
} from 'react';
import { makeStyles } from '@material-ui/core/styles';
import { AudioButton } from '../../Playback/AudioButton';
import PlayCircleOutlineIcon from '@material-ui/icons/PlayCircleOutline';
import PauseCircleOutlineIcon from '@material-ui/icons/PauseCircleOutline';
import IconButton from '@material-ui/core/IconButton';
import Box from '@material-ui/core/Box';

interface Props {
  url: string;
  currentlyPlayingRef: HTMLAudioElement | null;
  setCurrentlyPlayingRef: React.Dispatch<
    React.SetStateAction<HTMLAudioElement | null>
  >;
}

const useStyles = makeStyles(theme => ({
  playPause: { cursor: 'pointer' },
}));

export const ResultPlaybackCell = (props: Props) => {
  const classes = useStyles();

  const [isPlaying, setIsPlaying] = useState(false);
  const { url, currentlyPlayingRef, setCurrentlyPlayingRef } = props;
  const audioElementRef = useRef<HTMLAudioElement | null>(null);

  const playPauseClickHandler = () => {
    isPlaying ? pauseEvent() : playEvent();
  };

  const playEvent = useCallback(() => {
    if (audioElementRef.current) {
      audioElementRef.current.play();
      setCurrentlyPlayingRef(audioElementRef.current);
      setIsPlaying(true);
    }
  }, [setIsPlaying, setCurrentlyPlayingRef]);

  const pauseEvent = useCallback(() => {
    if (audioElementRef.current) {
      audioElementRef.current.pause();
      setIsPlaying(false);
    }
  }, [setIsPlaying]);

  const PlayPauseIcon = useMemo(() => {
    return (
      <>{isPlaying ? <PauseCircleOutlineIcon /> : <PlayCircleOutlineIcon />}</>
    );
  }, [isPlaying]);

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
    <div>
      <AudioButton audioElementRef={audioElementRef} url={url} />
      <div
        className="play-pause-button-wrapper"
        onClick={playPauseClickHandler}
      >
        <IconButton
          size="small"
          className={classes.playPause}
          edge="start"
          aria-label="play"
        >
          {PlayPauseIcon}
        </IconButton>
      </div>
    </div>
  );
};
