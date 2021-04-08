/**
 *
 * App
 *
 * This component is the skeleton around the actual pages, and should only
 * contain code that should be seen on all pages. (e.g. navigation bar)
 */

import * as React from 'react';
import { Helmet } from 'react-helmet-async';
import { Switch, Route, BrowserRouter } from 'react-router-dom';

import { unstable_createMuiStrictModeTheme as createMuiTheme } from '@material-ui/core';
import Typography from '@material-ui/core/Typography';
import { ThemeProvider } from '@material-ui/core/styles';
import makeStyles from '@material-ui/core/styles/makeStyles';
import CssBaseline from '@material-ui/core/CssBaseline';
import '@fontsource/roboto/300.css';
import '@fontsource/roboto/400.css';
import '@fontsource/roboto/500.css';
import '@fontsource/roboto/700.css';

import { Player } from './pages/Player/Loadable';
import { NotFoundPage } from './components/NotFoundPage/Loadable';
import { useTranslation } from 'react-i18next';

import Copyright from './components/Copyright';
import { GlobalStyle } from 'styles/global-styles';

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

const useStyles = makeStyles(theme => ({
  footer: {
    backgroundColor: theme.palette.background.paper,
    padding: theme.spacing(2),
  },
}));

export function App() {
  const classes = useStyles();
  const { i18n } = useTranslation();
  return (
    <>
      <BrowserRouter>
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
          <main>
            <Switch>
              {/* <Route exact path="/" component={HomePage} /> */}
              <Route exact path="/" component={Player} />
              <Route component={NotFoundPage} />
            </Switch>
          </main>
        </ThemeProvider>
      </BrowserRouter>
      <footer className={classes.footer}>
        <Typography variant="h6" align="center" gutterBottom>
          Footer
        </Typography>
        <Typography
          variant="subtitle1"
          align="center"
          color="textSecondary"
          component="p"
        >
          Something here to give the footer a purpose!
        </Typography>
        <Copyright />
      </footer>
    </>
  );
}
