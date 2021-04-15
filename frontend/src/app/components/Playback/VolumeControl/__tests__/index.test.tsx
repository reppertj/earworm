import * as React from 'react';
import { render } from '@testing-library/react';

import { VolumeControl } from '..';

describe('<VolumeControl  />', () => {
  it('should match snapshot', () => {
    const loadingIndicator = render(<VolumeControl />);
    expect(loadingIndicator.container.firstChild).toMatchSnapshot();
  });
});
