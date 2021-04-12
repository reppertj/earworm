import * as React from 'react';
import { Helmet } from 'react-helmet-async';
import makeStyles from '@material-ui/core/styles/makeStyles';
import AppBar from '@material-ui/core/AppBar';
import Toolbar from '@material-ui/core/Toolbar';
import LibraryMusicOutlinedIcon from '@material-ui/icons/LibraryMusicOutlined';
import Typography from '@material-ui/core/Typography';
import MusicSearchForm from '../../components/SearchForm';
import Copyright from '../../components/Copyright';

const useStyles = makeStyles(theme => ({
  icon: {
    marginRight: theme.spacing(2),
  },
  heroContent: {
    backgroundColor: theme.palette.background.paper,
    padding: theme.spacing(8, 0, 6),
  },
  footer: {
    backgroundColor: theme.palette.background.paper,
    padding: theme.spacing(2),
  },
}));

export function HomePage() {
  const classes = useStyles();

  return (
    <>
      <Helmet>
        <title>Earworm</title>
        <meta
          name="description"
          content="Search for royalty-free music by sonic similarity"
        />
      </Helmet>
      <main>
        <AppBar position="relative">
          <Toolbar>
            <LibraryMusicOutlinedIcon className={classes.icon} />
            <Typography variant="h6" color="inherit" noWrap>
              Earworm
            </Typography>
          </Toolbar>
        </AppBar>
        <MusicSearchForm />
      </main>
      <footer className={classes.footer}>
        <Typography variant="h6" align="center" gutterBottom>
          Footer
        </Typography>
        <Typography
          variant="subtitle1"
          align="center"
          color="textSecondary"
          component="p"
        >
          Something here to give the footer a purpose!
        </Typography>
        <Copyright />
      </footer>
    </>
  );
}
