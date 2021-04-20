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
import {
  DataGrid,
  GridColDef,
  GridToolbarContainer,
} from '@material-ui/data-grid';
import Link from '@material-ui/core/Link';
import LinkIcon from '@material-ui/icons/Link';
import Box from '@material-ui/core/Box';
import { ShareButton } from '../ShareButton';
import { ResultPlaybackCell } from '../ResultPlaybackCell';

function ShareToolbar() {
  return (
    <GridToolbarContainer>
      <ShareButton />
    </GridToolbarContainer>
  );
}

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

  const columns: GridColDef[] = [
    {
      field: 'percentMatch',
      headerName: '% Match',
      disableColumnMenu: true,
      filterable: false,
      width: 130,
    },
    {
      field: 'previewUrl',
      headerName: 'Preview',
      width: 50,
      sortable: false,
      filterable: false,
      disableColumnMenu: true,
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
    { field: 'artist', headerName: 'Artist', width: 150 },
    {
      field: 'provider',
      headerName: 'Source',
      flex: 4,
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
      valueGetter: params => params.row.license.name,
      sortComparator: (v1, v2, param1, param2) =>
        (param1.value as string).localeCompare(param2.value as string),
      renderCell: params => {
        console.log(params);
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

  const rows = searchResults.map((data, idx) => ({
    ...data,
    id: searchResultsIds[idx],
  }));

  return (
    <>
      {searchResults.length > 0 && (
        <>
          <div style={{ display: 'flex', width: '100%' }}>
            <div style={{ flexGrow: 1 }}>
              <DataGrid
                autoHeight
                pagination
                pageSize={25}
                rowsPerPageOptions={[]}
                components={{ Toolbar: ShareToolbar }}
                rows={rows}
                columns={columns}
                loading={isLoading}
                sortModel={[{ field: 'percentMatch', sort: 'desc' }]}
              />
            </div>
          </div>
        </>
      )}
    </>
  );
}
