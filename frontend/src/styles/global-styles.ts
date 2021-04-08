import { createGlobalStyle } from 'styled-components';

export const GlobalStyle = createGlobalStyle`
  region.wavesurfer-region::before {
    content: attr(data-id);
    color: rgba(0, 0, 0, 0.8);
    background-color: rgba(0, 0, 0, 0.125);
    padding-left: 1px;
    margin-left: 2px;
    margin-top: 2px;
  }
`;
