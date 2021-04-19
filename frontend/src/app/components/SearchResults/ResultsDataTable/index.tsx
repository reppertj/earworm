/**
 *
 * ResultsDataTable
 *
 */
import React, { useState } from 'react';
import { useSelector } from 'react-redux';
import {
  selectAllSearchResults,
  selectSearchResultsIds,
} from '../slice/selectors';
import { DataGrid, ColDef } from '@material-ui/data-grid';
import Link from '@material-ui/core/Link';
import LinkIcon from '@material-ui/icons/Link';
import Box from '@material-ui/core/Box';
import { ResultPlaybackCell } from '../ResultPlaybackCell';

interface Props {
  isLoading: boolean;
  isSuccess: boolean;
}

export function ResultsDataTable(props: Props) {
  const { isLoading, isSuccess } = props;
  const [
    currentlyPlayingRef,
    setCurrentlyPlayingRef,
  ] = useState<HTMLAudioElement | null>(null);
  const searchResults = useSelector(selectAllSearchResults);
  const searchResultsIds = useSelector(selectSearchResultsIds);

  const columns: ColDef[] = [
    { field: 'percentMatch', headerName: '% Match', width: 120 },
    {
      field: 'previewUrl',
      headerName: 'Preview',
      width: 50,
      sortable: false,
      renderHeader: () => <div> </div>,
      renderCell: params => (
        <Box padding={0} margin={0}>
          <ResultPlaybackCell
            url={params.value as string}
            currentlyPlayingRef={currentlyPlayingRef}
            setCurrentlyPlayingRef={setCurrentlyPlayingRef}
          />
        </Box>
      ),
    },
    {
      field: 'title',
      headerName: 'Title',
      width: 230,
      sortComparator: (v1, v2, param1, param2) =>
        (param1.value as any).name.localeCompare((param2.value as any).name),
      renderCell: params =>
        params.value ? (
          <Link
            target="_blank"
            rel="noopener"
            color="secondary"
            href={(params.value as any).url}
          >
            {(params.value as any).name} {<LinkIcon fontSize="small" />}
          </Link>
        ) : (
          <></>
        ),
    },
    { field: 'artist', headerName: 'Artist', width: 150 },
    {
      field: 'provider',
      headerName: 'Website',
      width: 150,
      sortComparator: (v1, v2, param1, param2) =>
        (param1.value as any).name.localeCompare((param2.value as any).name),
      renderCell: params =>
        params.value ? (
          <Link target="_blank" rel="noopener" href={(params.value as any).url}>
            {(params.value as any).name}
          </Link>
        ) : (
          <></>
        ),
    },
    {
      field: 'license',
      headerName: 'License',
      width: 150,
      sortComparator: (v1, v2, param1, param2) =>
        (param1.value as any).name.localeCompare((param2.value as any).name),
      renderCell: params =>
        params.value ? (
          <Link target="_blank" rel="noopener" href={(params.value as any).url}>
            {(params.value as any).name}
          </Link>
        ) : (
          <></>
        ),
    },
  ];

  const rows = searchResults.map((data, idx) => ({
    ...data,
    id: searchResultsIds[idx],
  }));

  return (
    <>
      {searchResults.length > 0 && (
        <>
          <div style={{ display: 'flex', height: 800, width: '100%' }}>
            <div style={{ flexGrow: 1 }}>
              <DataGrid
                rows={rows}
                columns={columns}
                loading={isLoading}
                hideFooterPagination={true}
                sortModel={[{ field: 'percentMatch', sort: 'desc' }]}
              />
            </div>
          </div>
          {/* <div style={{ height: 800, width: '100%' }}>
        </div> */}
        </>
      )}
    </>
  );
}
