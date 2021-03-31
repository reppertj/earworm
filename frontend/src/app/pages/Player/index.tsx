import * as React from 'react';
import { Helmet } from 'react-helmet-async';
import { InputWindow } from '../../components/uploader';

export function Player() {
  const sources = [
    'https://files.freemusicarchive.org/storage-freemusicarchive-org/music/Music_for_Video/Little_Glass_Men/Debut_EP/Little_Glass_Men_-_02_-_Golden.mp3',
    'https://files.freemusicarchive.org/storage-freemusicarchive-org/tracks/7ZStaYrFwp3U7liEc6YGBb23GI3Uh5weBxU4T1VT.mp3',
  ];

  return (
    <>
      <Helmet>
        <title>Home Page</title>
        <meta name="description" content="A Boilerplate application homepage" />
      </Helmet>
      <InputWindow />
    </>
  );
}
