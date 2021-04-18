/**
 *
 * ResultsDataTable
 *
 */
import * as React from 'react';
import { useSelector } from 'react-redux';
import { selectAllSearchResults } from '../slice/selectors';
// import DataGrid from '@material-ui/'

interface Props {
  isLoading: boolean;
  isSuccess: boolean;
}

export function ResultsDataTable(props: Props) {
  const { isLoading, isSuccess } = props;
  const searchResults = useSelector(selectAllSearchResults);
  return <div>{isSuccess && JSON.stringify(searchResults)}</div>;
}
