/**
 * index.tsx
 *
 * This is the entry file for the application, only setup and boilerplate
 * code.
 */

import 'react-app-polyfill/ie11';
import 'react-app-polyfill/stable';

import * as React from 'react';
import * as ReactDOM from 'react-dom';
import { Provider } from 'react-redux';

// Use consistent styling
import 'sanitize.css/sanitize.css';

// Import root app
import { App } from 'app';

import { configureAppStore, rootSaga } from 'store/configureStore';

import reportWebVitals from 'reportWebVitals';

// Initialize languages
import './locales/i18n';

// GA
import GA4React from 'ga-4-react';
const gaId = process.env.REACT_APP_GA_MEASUREMENT_ID;
console.log(gaId);
const ga4react = new GA4React(gaId ?? '');

const store = configureAppStore();
store.runSaga(rootSaga);
const MOUNT_NODE = document.getElementById('root') as HTMLElement;

(async () => {
  try {
    await ga4react.initialize();
    console.log(ga4react);
  } catch (e) {
    console.log(e);
  }
  ReactDOM.render(
    <Provider store={store}>
      <React.StrictMode>
        <App />
      </React.StrictMode>
    </Provider>,
    MOUNT_NODE,
  );
})();

// Hot reloadable translation json files
if (module.hot) {
  module.hot.accept(['./locales/i18n'], () => {
    // No need to render the App again because i18next works with the hooks
  });
}

// If you want to start measuring performance in your app, pass a function
// to log results (for example: reportWebVitals(console.log))
// or send to an analytics endpoint. Learn more: https://bit.ly/CRA-vitals
reportWebVitals();
