/**
 *
 * ResultsWindow
 *
 */
import React, { useState } from 'react';
import { useQuery } from 'react-query';
import { postEmbeddingsSearch, cancelTokenSource } from '../../../utils/api';

import {
  selectAllEmbeddings,
  selectNumEmbeddings,
} from 'app/components/SearchInputs/SearchMeta/slice/selectors';
import { Provider, useDispatch, useSelector } from 'react-redux';
import { useEmbeddingsSlice } from 'app/components/SearchInputs/SearchMeta/slice';
import { useErrorSlice } from 'app/components/Error/slice';
import { useSearchResultsSlice } from '../slice';
import { nanoid } from '@reduxjs/toolkit';
import { ResultsDataTable } from '../ResultsDataTable';

interface Props {}

export function ResultsWindow(props: Props) {
  const [page, setPage] = useState(0);
  const [results, setResults] = useState<any>(undefined);
  const numEmbeddings = useSelector(selectNumEmbeddings);
  const allEmbeddings = useSelector(selectAllEmbeddings);
  const dispatch = useDispatch();
  const { actions: errorActions } = useErrorSlice();
  const { actions: resultsActions } = useSearchResultsSlice();
  const { actions: embeddingActions } = useEmbeddingsSlice();

  const doSearch = numEmbeddings > 0;

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
  } else if (!isLoading && data) {
    const flattened = data.data.map(result => ({
      id: nanoid(),
      title: result.title,
      artist: result.artist,
      url: result.url,
      licenseName: result.license.name,
      licenseUrl: result.license.url,
      providerName: result.provider.name,
      providerUrl: result.provider.url,
      previewUrl: result.temp_preview_url,
      percentMatch: result.percent_match,
    }));
    dispatch(resultsActions.searchResultsAllSet(flattened));
  }
  dispatch(embeddingActions.embeddingsAllRemoved);

  return (
    <div>
      {isLoading ? (
        'Loading...'
      ) : (
        <ResultsDataTable isLoading={isLoading} isSuccess={isSuccess} />
      )}
    </div>
  );
}
