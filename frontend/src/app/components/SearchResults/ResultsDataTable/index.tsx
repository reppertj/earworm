/**
 *
 * ResultsDataTable
 *
 */
import React, { useState, useMemo } from 'react';
import { useSelector } from 'react-redux';
import {
  selectAllSearchResults,
  selectSearchResultsIds,
} from '../slice/selectors';
import { DataGrid, GridColDef } from '@material-ui/data-grid';
import Link from '@material-ui/core/Link';
import LinkIcon from '@material-ui/icons/Link';
import Box from '@material-ui/core/Box';
import useMediaQuery from '@material-ui/core/useMediaQuery';
import { ResultPlaybackCell } from '../ResultPlaybackCell';

interface Props {}

export function ResultsDataTable(props: Props) {
  const [
    currentlyPlayingRef,
    setCurrentlyPlayingRef,
  ] = useState<HTMLAudioElement | null>(null);
  const [sortModelChanged, setSortModelChanged] = useState<boolean>(false);
  const bigger = useMediaQuery('(min-width:600px)');
  const searchResults = useSelector(selectAllSearchResults);
  const searchResultsIds = useSelector(selectSearchResultsIds);

  const columns: GridColDef[] = [
    {
      field: 'percentMatch',
      headerName: '% Match',
      disableColumnMenu: true,
      filterable: false,
      disableClickEventBubbling: true,
      width: bigger ? 130 : undefined,
      flex: bigger ? undefined : 0.7,
    },
    {
      field: 'previewUrl',
      headerName: 'Preview',
      width: 50,
      // sortable: false,
      // filterable: false,
      disableColumnMenu: true,
      disableClickEventBubbling: true,
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
      flex: 6,
      disableClickEventBubbling: true,
      valueGetter: params => params.row.title.name,
      sortComparator: (v1, v2, param1, param2) =>
        (param1.value as string).localeCompare(param2.value as string),
      renderCell: params =>
        params.value ? (
          <Link
            target="_blank"
            rel="noopener"
            color="secondary"
            href={params.row.title.url}
          >
            {params.getValue('title')} {<LinkIcon fontSize="small" />}
          </Link>
        ) : (
          <></>
        ),
    },
    {
      field: 'artist',
      disableClickEventBubbling: true,
      headerName: 'Artist',
      width: 150,
    },
    {
      field: 'provider',
      headerName: 'Source',
      flex: 4,
      disableClickEventBubbling: true,
      valueGetter: params => params.row.provider.name,
      sortComparator: (v1, v2, param1, param2) =>
        (param1.value as string).localeCompare(param2.value as string),
      renderCell: params => {
        return params.value ? (
          <Link target="_blank" rel="noopener" href={params.row.provider.url}>
            {params.getValue('provider')}
          </Link>
        ) : (
          <></>
        );
      },
    },
    {
      field: 'license',
      headerName: 'License',
      flex: 3,
      disableClickEventBubbling: true,
      valueGetter: params => params.row.license.name,
      sortComparator: (v1, v2, param1, param2) =>
        (param1.value as string).localeCompare(param2.value as string),
      renderCell: params => {
        return params.value ? (
          <Link target="_blank" rel="noopener" href={params.row.license.url}>
            {params.getValue('license')}
          </Link>
        ) : (
          <></>
        );
      },
    },
  ];

  const handleSortModelChange = () => {
    setSortModelChanged(true);
  };

  const rows = useMemo(() => {
    setSortModelChanged(false);
    return searchResults.map((data, idx) => ({
      ...data,
      id: searchResultsIds[idx],
    }));
  }, [searchResults, searchResultsIds]);

  return (
    <>
      {searchResults.length > 0 && (
        <>
          <div style={{ display: 'flex', height: '100%', width: '100%' }}>
            <div style={{ flexGrow: 1 }}>
              <DataGrid
                autoHeight
                pagination
                hideFooterSelectedRowCount
                pageSize={25}
                rowsPerPageOptions={[]}
                rows={rows}
                columns={columns}
                onSortModelChange={handleSortModelChange}
                sortModel={
                  sortModelChanged
                    ? undefined
                    : [{ field: 'percentMatch', sort: 'desc' }]
                }
              />
            </div>
          </div>
        </>
      )}
    </>
  );
}
