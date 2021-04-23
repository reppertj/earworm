import React from 'react';
import Typography from '@material-ui/core/Typography';
import Link from '@material-ui/core/Link';

export default function Copyright(props) {
  return (
    <Typography variant="body2" color="textSecondary" align="center">
      {'By '}
      <Link
        color="inherit"
        target="_blank"
        rel="noopner"
        href="https://www.justinreppert.com/"
      >
        Justin Reppert
      </Link>{' '}
      {new Date().getFullYear()}
      {'.'}
    </Typography>
  );
}
