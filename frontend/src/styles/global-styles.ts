import { createGlobalStyle } from 'styled-components';

/* stylelint-disable selector-type-no-unknown */
export const GlobalStyle = createGlobalStyle`
  region.wavesurfer-region::before {
    content: attr(data-region-label);
    color: rgba(0, 0, 0, 0.9);
    background-color: rgba(248, 248, 255, 0.8);
    padding-left: 1px;
    margin-left: 2px;
    margin-top: 2px;
  }
`;
/* stylelint-enable selector-type-no-unknown */
