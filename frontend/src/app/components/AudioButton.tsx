import React, { useRef } from 'react';

interface Props {
  audioElementRef: React.MutableRefObject<HTMLAudioElement | null>;
  url: string | undefined;
}

export default function AudioButton(props: Props) {
  return (
    <audio
      ref={props.audioElementRef}
      src={props.url}
      crossOrigin={'anonymous'}
    ></audio>
  );
}
