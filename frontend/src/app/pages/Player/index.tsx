import * as React from 'react';
import { Helmet } from 'react-helmet-async';
import MusicSearchForm from '../../components/SearchForm';

export function Player() {
  return (
    <>
      <Helmet>
        <title>Home Page</title>
        <meta name="description" content="A Boilerplate application homepage" />
      </Helmet>
      <MusicSearchForm />
    </>
  );
}
