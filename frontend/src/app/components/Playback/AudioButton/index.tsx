/**
 *
 * AudioButton
 *
 */
import React, { memo } from 'react';

interface Props {
  audioElementRef: React.MutableRefObject<HTMLAudioElement | null>;
  url: string | undefined;
}

export const AudioButton = memo((props: Props) => {
  return (
    <audio
      ref={props.audioElementRef}
      src={props.url}
      crossOrigin={'anonymous'}
    ></audio>
  );
});
