/**
 * Asynchronously loads the component for HomePage
 */

import { lazyLoad } from 'utils/loadable';

export const Player = lazyLoad(
  () => import('./index'),
  module => module.Player,
);
