/**
 *
 * Asynchronously loads the component for ResultsDataTable
 *
 */

import { lazyLoad } from 'utils/loadable';

export const ResultsDataTable = lazyLoad(
  () => import('./index'),
  module => module.ResultsDataTable,
);
