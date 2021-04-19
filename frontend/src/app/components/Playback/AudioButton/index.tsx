/**
 *
 * AudioButton
 *
 */
import React from 'react';
import { useSelector } from 'react-redux';
import { selectVolume } from '../volumeSlice/selectors';

interface Props {
  audioElementRef: React.MutableRefObject<HTMLAudioElement | null>;
  url: string | undefined;
}

export const AudioButton = (props: Props) => {
  const { value: volume } = useSelector(selectVolume);
  if (props.audioElementRef.current) {
    props.audioElementRef.current.volume = volume;
  }
  return (
    <audio
      ref={props.audioElementRef}
      src={props.url}
      crossOrigin={'anonymous'}
      preload="none"
    ></audio>
  );
};
