import React from 'react';
import LinearProgress from '@material-ui/core/LinearProgress';
import Box from '@material-ui/core/Box';
import Typography from '@material-ui/core/Typography';

interface SearchProgressProps {
  searching: boolean;
}

export default function SearchProgress(props: SearchProgressProps) {
  const { searching } = props;

  return (
    <>
      {searching && (
        <Box display="flex" alignItems="center">
          <Box width="100%" mr={1}>
            {<LinearProgress />}
          </Box>
        </Box>
      )}
    </>
  );
}
