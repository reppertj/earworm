/**
 *
 * ResultsWindow
 *
 */
import React, { useState } from 'react';
import { useQuery } from 'react-query';
import { postEmbeddingsSearch } from '../../../utils/api';

import { selectAllEmbeddings } from 'app/components/SearchInputs/SearchMeta/slice/selectors';
import { useDispatch, useSelector } from 'react-redux';
import { useEmbeddingsSlice } from 'app/components/SearchInputs/SearchMeta/slice';
import { useErrorSlice } from 'app/components/Error/slice';
import { useSearchResultsSlice } from '../slice';
import { nanoid } from '@reduxjs/toolkit';
import { ResultsDataTable } from '../ResultsDataTable';
import Paper from '@material-ui/core/Paper';
import Container from '@material-ui/core/Container';
import Grid from '@material-ui/core/Grid';
import Box from '@material-ui/core/Box';
import CircularProgress from '@material-ui/core/CircularProgress';
import Typography from '@material-ui/core/Typography';

interface Props {}

export function ResultsWindow(props: Props) {
  // eslint-disable-next-line @typescript-eslint/no-unused-vars
  const [page, setPage] = useState(0);
  const allEmbeddings = useSelector(selectAllEmbeddings);
  const dispatch = useDispatch();
  const { actions: errorActions } = useErrorSlice();
  const { actions: resultsActions } = useSearchResultsSlice();
  const { actions: embeddingActions } = useEmbeddingsSlice();

  const toSearch = allEmbeddings[allEmbeddings.length - 1];

  const embeddings = toSearch?.embeddings;

  const { isLoading, isError, data, error } = useQuery(
    ['embeddingsSearch', embeddings, page],
    postEmbeddingsSearch,
    {
      enabled: !!embeddings,
      staleTime: 1000 * 60 * 15,
    },
  );
  if (isError) {
    console.log(error);
    dispatch(
      errorActions.errorAdded({
        id: nanoid(),
        error: true,
        message: 'Error communicating with the server.',
      }),
    );
  } else if (!isLoading && data) {
    const flattened = data.data.map(result => ({
      id: nanoid(),
      title: {
        name: result.title,
        url: result.url,
      },
      artist: result.artist,
      license: {
        name: result.license.name,
        url: result.license.url,
      },
      provider: {
        name: result.provider.name,
        url: result.provider.url,
      },
      previewUrl: result.temp_preview_url,
      percentMatch: result.percent_match,
    }));
    dispatch(resultsActions.searchResultsAllSet(flattened));
  }
  dispatch(embeddingActions.embeddingsAllRemoved);

  if (isLoading) {
    return (
      <Grid container direction="column" alignItems="center" justify="center">
        <Grid item>
          <Typography>Waiting for matching results</Typography>
        </Grid>
        <Box pt={2} pb={20}>
          <Grid item>
            <CircularProgress size={40} />
          </Grid>
        </Box>
      </Grid>
    );
  } else {
    return (
      <div>
        <Container>
          <Paper>
            <ResultsDataTable />
          </Paper>
        </Container>
      </div>
    );
  }
}
