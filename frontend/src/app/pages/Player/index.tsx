import * as React from 'react';
import { Helmet } from 'react-helmet-async';
import makeStyles from '@material-ui/core/styles/makeStyles';
import AppBar from '@material-ui/core/AppBar';
import Toolbar from '@material-ui/core/Toolbar';
import LibraryMusicOutlinedIcon from '@material-ui/icons/LibraryMusicOutlined';
import Typography from '@material-ui/core/Typography';
import MusicSearchForm from '../../components/SearchForm';

const useStyles = makeStyles(theme => ({
  icon: {
    marginRight: theme.spacing(2),
  },
  heroContent: {
    backgroundColor: theme.palette.background.paper,
    padding: theme.spacing(8, 0, 6),
  },
}));

export function Player() {
  const classes = useStyles();

  return (
    <>
      <Helmet>
        <title>Home Page</title>
        <meta name="description" content="A Boilerplate application homepage" />
      </Helmet>
      <AppBar position="relative">
        <Toolbar>
          <LibraryMusicOutlinedIcon className={classes.icon} />
          <Typography variant="h6" color="inherit" noWrap>
            Earworm
          </Typography>
        </Toolbar>
      </AppBar>
      <MusicSearchForm />
    </>
  );
}
