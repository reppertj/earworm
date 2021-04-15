/**
 *
 * AppBar
 *
 */
import React, { memo } from 'react';
import MuiAppBar from '@material-ui/core/AppBar';
import makeStyles from '@material-ui/core/styles/makeStyles';
import Typography from '@material-ui/core/Typography';
import Toolbar from '@material-ui/core/Toolbar';
import LibraryMusicOutlinedIcon from '@material-ui/icons/LibraryMusicOutlined';

import { VolumeControl } from '../Playback/VolumeControl';

const useStyles = makeStyles(theme => ({
  root: {
    flexGrow: 1,
  },
  title: {
    flexGrow: 1,
    display: 'none',
    [theme.breakpoints.up('sm')]: {
      display: 'block',
    },
  },
  icon: {
    marginRight: theme.spacing(2),
  },
  volume: {
    position: 'relative',
    marginLeft: 0,
    width: '100%',
    [theme.breakpoints.up('sm')]: {
      marginLeft: theme.spacing(1),
      width: 'auto',
    },
  },
}));

interface Props {}

const AppBar = memo((props: Props) => {
  const classes = useStyles();

  return (
    <div className={classes.root}>
      <MuiAppBar position="relative">
        <Toolbar>
          <LibraryMusicOutlinedIcon className={classes.icon} />
          <Typography variant="h6" color="inherit" noWrap>
            Earworm
          </Typography>
          <div className={classes.volume}>
            <VolumeControl />
          </div>
        </Toolbar>
      </MuiAppBar>
    </div>
  );
});

export default AppBar;
