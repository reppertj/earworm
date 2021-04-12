/**
 *
 * App
 *
 * This component is the skeleton around the actual pages, and should only
 * contain code that should be seen on all pages. (e.g. navigation bar)
 */

import * as React from 'react';
import { Helmet } from 'react-helmet-async';
import { ConnectedRouter } from 'connected-react-router';
import { Switch, Route } from 'react-router-dom';

import { unstable_createMuiStrictModeTheme as createMuiTheme } from '@material-ui/core';
import { ThemeProvider } from '@material-ui/core/styles';
import CssBaseline from '@material-ui/core/CssBaseline';
import '@fontsource/roboto/300.css';
import '@fontsource/roboto/400.css';
import '@fontsource/roboto/500.css';
import '@fontsource/roboto/700.css';

import { HomePage } from './pages/HomePage/Loadable';
import { Admin } from './pages/Admin';
import { Login } from './pages/Login/Loadable';
import { NotFoundPage } from './components/NotFoundPage/Loadable';
import { useTranslation } from 'react-i18next';

import { Protected } from './components/Protected';
import { GlobalStyle } from 'styles/global-styles';
import { logout } from './utils/auth';
import { PrivateRoute } from './components/PrivateRoute';
import history from './utils/history';

const theme = createMuiTheme({
  palette: {
    primary: {
      main: '#78909c',
    },
    secondary: {
      main: '#ffca28',
    },
  },
});

export function App() {
  const { i18n } = useTranslation();
  console.log('history before', history);
  return (
    <>
      <ConnectedRouter history={history}>
        <CssBaseline />
        <GlobalStyle />
        <Helmet
          titleTemplate="Earworm"
          defaultTitle="Earworm"
          htmlAttributes={{ lang: i18n.language }}
        >
          <meta
            name="description"
            content="Search for royalty-free music licensed for commercial use by sonic similarity."
          />
          <meta
            name="viewport"
            content="minimum-scale=1, initial-scale=1, width=device-width"
          />
        </Helmet>

        <ThemeProvider theme={theme}>
          <Switch>
            <Route path="/admin" render={() => <Admin />} />
            <Route path="/login" component={Login} />
            <Route
              path="/logout"
              render={() => {
                logout();
                history.push('/');
                return null;
              }}
            />
            <PrivateRoute path="/protected" component={Protected} />
            <Route exact path="/" component={HomePage} />
            <Route component={NotFoundPage} />
          </Switch>
        </ThemeProvider>
      </ConnectedRouter>
    </>
  );
}
