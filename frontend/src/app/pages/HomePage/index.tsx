import * as React from 'react';
import { Helmet } from 'react-helmet-async';
import makeStyles from '@material-ui/core/styles/makeStyles';
import Typography from '@material-ui/core/Typography';

import AppBar from '../../components/AppBar';
import MusicSearchForm from '../../components/SearchInputs/SearchForm';
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
        <AppBar />
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
