import React from 'react';

export default function AudioHandler(props) {
  const audioCtx = new (window.AudioContext || window.webkitAudioContext)();
}
