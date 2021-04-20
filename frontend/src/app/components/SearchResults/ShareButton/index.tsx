/**
 *
 * ShareButton
 *
 */

import React from 'react';

import ShareIcon from '@material-ui/icons/Share';
import Button from '@material-ui/core/Button';

interface Props {}

export function ShareButton(props: Props) {
  return (
    <div>
      <Button startIcon={<ShareIcon fontSize="small" />}>Share</Button>
    </div>
  );
}
