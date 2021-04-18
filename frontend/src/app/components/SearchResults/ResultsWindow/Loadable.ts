/**
 *
 * Asynchronously loads the component for ResultsWindow
 *
 */

import { lazyLoad } from 'utils/loadable';

export const ResultsWindow = lazyLoad(
  () => import('./index'),
  module => module.ResultsWindow,
);
