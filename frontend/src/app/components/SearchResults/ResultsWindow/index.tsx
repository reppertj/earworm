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

interface Props {}

export function ResultsWindow(props: Props) {
  const [page, setPage] = useState(0);
  const allEmbeddings = useSelector(selectAllEmbeddings);
  const dispatch = useDispatch();
  const { actions: errorActions } = useErrorSlice();
  const { actions: resultsActions } = useSearchResultsSlice();
  const { actions: embeddingActions } = useEmbeddingsSlice();

  const toSearch = allEmbeddings[allEmbeddings.length - 1];
  console.log('searching with', toSearch);

  const embeddings = toSearch?.embeddings;

  const { isLoading, isError, isSuccess, data, error } = useQuery(
    ['embeddingsSearch', embeddings, page],
    postEmbeddingsSearch,
    {
      enabled: !!embeddings,
      staleTime: 1000 * 60 * 5,
    },
  );
  if (isError) {
    // TODO: error handling on response from query
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

  return (
    <div>
      <Container>
        <Paper>
          <ResultsDataTable isLoading={isLoading} isSuccess={isSuccess} />
        </Paper>
      </Container>
    </div>
  );
}
