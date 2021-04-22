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
const ga4react = new GA4React(gaId ?? '');

// Redux/Saga
const store = configureAppStore();
store.runSaga(rootSaga);
const MOUNT_NODE = document.getElementById('root') as HTMLElement;

(async () => {
  try {
    await ga4react.initialize();
    ga4react.gtag('set', { cookie_flags: 'SameSite=None;Secure' });
  } catch (e) {}
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

// Measure performance experienced by end users
function vitalsToAnalytics({ id, name, value }) {
  ga4react.gtag('event', name, {
    event_category: 'Web Vitals',
    event_label: id,
    value: Math.round(name === 'CLS' ? value * 1000 : value),
    non_interaction: true,
  });
}

reportWebVitals(vitalsToAnalytics);
