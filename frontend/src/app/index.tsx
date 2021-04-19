/**
 *
 * App
 *
 * This component is the skeleton around the actual pages, and should only
 * contain code that should be seen on all pages. (e.g. navigation bar)
 */

import * as React from 'react';
import { Helmet } from 'react-helmet';
import { Switch, Route, BrowserRouter } from 'react-router-dom';
import { QueryClient, QueryClientProvider } from 'react-query';

import { createMuiTheme } from '@material-ui/core';
import { ThemeProvider, StylesProvider } from '@material-ui/core/styles';
import CssBaseline from '@material-ui/core/CssBaseline';
import '@fontsource/roboto/300.css';
import '@fontsource/roboto/400.css';
import '@fontsource/roboto/500.css';
import '@fontsource/roboto/700.css';

import { HomePage } from './pages/HomePage/Loadable';
import { NotFoundPage } from './components/NotFoundPage/Loadable';
import { useTranslation } from 'react-i18next';

import { GlobalStyle } from 'styles/global-styles';
import { useGA4React } from 'ga-4-react';

const theme = createMuiTheme({
  palette: {
    primary: {
      main: '#8daa91',
    },
    secondary: {
      main: '#92140c',
    },
  },
});

const queryClient = new QueryClient();

export function App() {
  const gaId = process.env.REACT_APP_GA_MEASUREMENT_ID;
  try {
    const ga = useGA4React(gaId);
    console.log(ga);
  } catch (e) {
    console.log(e);
  }
  const { i18n } = useTranslation();
  return (
    <>
      <BrowserRouter>
        <CssBaseline />
        <StylesProvider injectFirst>
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
            <QueryClientProvider client={queryClient}>
              <Switch>
                <Route exact path="/" component={HomePage} />
                <Route component={NotFoundPage} />
              </Switch>
            </QueryClientProvider>
          </ThemeProvider>
        </StylesProvider>
      </BrowserRouter>
    </>
  );
}
