import * as React from 'react';
import { Helmet } from 'react-helmet-async';

import { fetchUtils, Admin as ReactAdmin, Resource } from 'react-admin';
import simpleRestProvider from 'ra-data-simple-rest';
import authProvider from './authProvider';

import { UserList } from '../../components/AdminResources/UserList';

import history from '../../utils/history';

const httpClient = (url: any, options: any) => {
  if (!options) {
    options = {};
  }
  if (!options.headers) {
    options.headers = new Headers({ Accept: 'application/json' });
  }
  const token = localStorage.getItem('token');
  options.headers.set('Authorization', `Bearer ${token}`);
  return fetchUtils.fetchJson(url, options);
};

const dataProvider = simpleRestProvider('api/v1', httpClient);

export function Admin() {
  return (
    <>
      <Helmet>
        <title>Earworm Admin</title>
        <meta name="description" content="Earworm admin functions" />
      </Helmet>
      <ReactAdmin
        dataProvider={dataProvider}
        authProvider={authProvider}
        history={history}
      >
        {(permissions: 'admin' | 'user') => [
          permissions === 'admin' ? (
            <Resource name="admin/users" list={UserList} />
          ) : null,
        ]}
      </ReactAdmin>
    </>
  );
}
