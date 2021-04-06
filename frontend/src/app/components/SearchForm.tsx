import React, {
  useCallback,
  useEffect,
  useMemo,
  useRef,
  useState,
} from 'react';

import { useDropzone, FileError } from 'react-dropzone';
import getChannelDataAndSampleRate, {
  FullAudioData,
} from '../utils/audioUtils';
import { UploadSurferPair } from './UploadSurfer';
import ErrorSnackBar from './ErrorSnackBar';
import SpotifySearchForm from './SpotifySearchForm';
import SpotifyAuthorizationButton from './SpotifyAuthorizationButton';
import { SpotifyChooser } from './SpotifyResultChooser';
import Worker from '../worker';

var inferenceWorker = new Worker();

interface RegionStart {
  id: number;
  seconds: number;
}

interface Source {
  id: number;
  regionStarts: RegionStart[];
  source?: string | File;
  channelData?: FullAudioData;
}

type Sources = Array<Source | null>;

const baseStyle = {
  flex: 1,
  display: 'flex',
  flexDirection: 'column' as const, // to make TS happy, not sure why it's only on this property
  padding: '20px',
  borderWidth: 2,
  borderRadius: 2,
  borderColor: '#eeeeee',
  borderStyle: 'dashed',
  backgroundColor: '#fafafa',
  color: '#bdbdbd',
  outline: 'none',
  transition: 'border .24s ease-in-out',
};

const activeStyle = {
  borderColor: '#2196f3',
};

const acceptStyle = {
  borderColor: '#00e676',
};

const rejectStyle = {
  borderColor: '#ff1744',
};

function prepareRegionInfo(
  sources: Array<Source>,
  regions: { 0: any[]; 1: any[] },
) {
  const startPoints: [Source, number][] = [];
  for (let i = 0; i < 2; i++) {
    for (let region of regions[i]) {
      startPoints.push([sources[i], region.start]);
    }
  }
  return startPoints;
}

async function getMaybeMultipleChannelData(sources: Sources): Promise<Sources> {
  const newSources = sources.slice();
  for (let i = 0; i < sources.length; i++) {
    const source = newSources[i];
    if (source && source.source) {
      const channelData = await getChannelDataAndSampleRate(source.source);
      newSources[i] = { channelData, ...source };
    }
  }
  return newSources;
}

function MusicSearchForm(props) {
  const [loggedIn, setLoggedIn] = useState(false);
  const [accessToken, setAccessToken] = useState<string | null>(null);
  const [tokenExpiryDate, setTokenExpiryDate] = useState<number | null>(null);
  const [hasError, setHasError] = useState({ error: false, message: '' });
  const [spotifyResults, setSpotifyResults] = useState<any>(null);
  const [chooserOpen, setChooserOpen] = useState(false);
  const [sources, setSources] = useState<Sources>([null, null]);
  const [filesLeft, setFilesLeft] = useState(2);

  const addSourceIfSpace = useCallback(
    (sourceData: string | File) => {
      let newSources = sources.slice();
      let openPosition = newSources.findIndex(elem => elem === null);
      if (openPosition === -1) {
        openPosition = newSources.length - 1;
      }
      newSources[openPosition] = {
        id: Date.now(),
        source: sourceData,
        regionStarts: [],
      };
      setSources(newSources);
      setFilesLeft(newSources.filter(source => source === null).length);
    },
    [sources],
  );

  const removeSource = useCallback(
    (sourcePosition: number) => {
      const newSources = sources.slice();
      if ([0, 1].includes(sourcePosition)) {
        newSources[sourcePosition] = null;
        setSources(newSources);
        setFilesLeft(newSources.filter(source => source === null).length);
      }
    },
    [sources],
  );

  const validateFilesLeft = () =>
    filesLeft > 0
      ? null
      : { message: 'Too many files', code: 'too-many-files' };

  const SpotifyElement = useMemo(() => {
    if (!loggedIn || (tokenExpiryDate && tokenExpiryDate < Date.now())) {
      return (
        <SpotifyAuthorizationButton
          loggedIn={loggedIn}
          setLoggedIn={setLoggedIn}
          setAccessToken={setAccessToken}
          setTokenExpiryDate={setTokenExpiryDate}
          setHasError={setHasError}
        />
      );
    } else {
      return (
        <SpotifySearchForm
          accessToken={accessToken}
          setLoggedIn={setLoggedIn}
          setHasError={setHasError}
          setSpotifyResults={setSpotifyResults}
        />
      );
    }
  }, [loggedIn, accessToken, tokenExpiryDate]);

  useEffect(() => {
    if (!chooserOpen && spotifyResults) {
      if (!spotifyResults.tracks) {
        setHasError({
          error: true,
          message:
            "Couldn't process the results from Spotify. Sorry about that.",
        });
      } else if (spotifyResults.tracks.items.length < 1) {
        setHasError({
          error: true,
          message: 'No results! Try searching for something else.',
        });
      } else {
        setChooserOpen(true);
      }
    }
  }, [spotifyResults, chooserOpen]);

  return (
    <>
      <UploadWindow
        sources={sources}
        setSources={setSources}
        removeSource={removeSource}
        addSourceIfSpace={addSourceIfSpace}
        filesLeft={filesLeft}
        setFilesLeft={setFilesLeft}
        SpotifyElement={SpotifyElement}
      />
      {hasError.error && (
        <ErrorSnackBar hasError={hasError} setHasError={setHasError} />
      )}
      {chooserOpen && (
        <SpotifyChooser
          spotifyResults={spotifyResults}
          setSpotifyResults={setSpotifyResults}
          open={chooserOpen}
          setOpen={setChooserOpen}
          addSourceIfSpace={addSourceIfSpace}
        />
      )}
    </>
  );
}

interface UploadWindowProps {
  sources: Sources;
  setSources: React.Dispatch<React.SetStateAction<Sources>>;
  filesLeft: number;
  setFilesLeft: React.Dispatch<React.SetStateAction<number>>;
  addSourceIfSpace: (sourceData: string | File) => void;
  removeSource: (sourcePosition: number) => void;
  SpotifyElement: JSX.Element;
}

function UploadWindow(props: UploadWindowProps) {
  const {
    sources,
    setSources,
    filesLeft,
    setFilesLeft,
    addSourceIfSpace,
    removeSource,
  } = props;

  const validateFilesLeft = () =>
    filesLeft > 0
      ? null
      : { message: 'Too many files', code: 'too-many-files' };

  const {
    acceptedFiles,
    open,
    getRootProps,
    getInputProps,
    isDragActive,
    isDragAccept,
    isDragReject,
  } = useDropzone({
    onDrop: acceptedFiles => {
      acceptedFiles.forEach(file => {
        addSourceIfSpace(file);
      });
    },
    accept: 'audio/*',
    noClick: true,
    noKeyboard: true,
    multiple: true,
    maxFiles: filesLeft,
    validator: validateFilesLeft,
  });

  // const {
  //   acceptedFiles: buttonAccepted,
  //   open,
  //   getRootProps: buttonRootProps,
  //   getInputProps: buttonInputProps,
  // } = useDropzone({
  //   onDrop: buttonAccepted => {
  //     buttonAccepted.map(file => setSources(sources.concat(file)));
  //   },
  //   accept: 'audio/*',
  //   noKeyboard: true,
  //   multiple: true,
  //   maxFiles: filesLeft,
  // });

  const style = useMemo(
    () => ({
      ...baseStyle,
      ...(isDragActive ? activeStyle : {}),
      ...(isDragAccept ? acceptStyle : {}),
      ...(isDragReject ? rejectStyle : {}),
    }),
    [isDragActive, isDragReject, isDragAccept],
  );

  const addRegionStart = useCallback(
    (sourceID, startTime) => {
      const newSources = sources.slice();
      const sourceIDX = newSources.findIndex(source => source?.id === sourceID);
      const regionID = Date.now();
      newSources[sourceIDX]?.regionStarts.push({
        id: regionID,
        seconds: startTime,
      });
      setSources(newSources);
      return regionID;
    },
    [sources, setSources],
  );

  const removeRegionStart = useCallback(
    (sourceID, regionID) => {
      const newSources = sources.slice();
      const sourceIDX = newSources.findIndex(source => source?.id === sourceID);
      const regionIDX = newSources[sourceIDX]?.regionStarts.findIndex(
        start => start.id === regionID,
      );
      if (regionIDX) {
        newSources[sourceIDX]?.regionStarts.splice(regionIDX, 1);
      }
      setSources(newSources);
    },
    [sources, setSources],
  );

  const updateRegionStart = useCallback(
    (sourceID, regionID, startTime) => {
      const newSources = sources.slice();
      const sourceIDX = newSources.findIndex(source => source?.id === sourceID);
      const regionIDX = newSources[sourceIDX]?.regionStarts.findIndex(
        start => start.id === regionID,
      );
      if (regionIDX !== undefined) {
        (newSources[sourceIDX] as Source).regionStarts[
          regionIDX
        ].seconds = startTime;
      }
      setSources(newSources);
    },
    [sources, setSources],
  );

  return (
    <React.Fragment>
      <div className="audioSearchWindow">
        <div {...getRootProps({ style })}>
          <input {...getInputProps()} />
          <button onClick={open} disabled={filesLeft < 1}>
            Search by mp3 ({filesLeft})
          </button>
          {props.SpotifyElement}
          <SearchWindow
            sources={sources}
            setSources={setSources}
            removeSource={removeSource}
            regionStartCallbacks={{
              removeRegionStart,
              addRegionStart,
              updateRegionStart,
            }}
          />
        </div>
      </div>
    </React.Fragment>
  );
}

function SearchWindow(props) {
  const [regions, setRegions] = useState({ 0: [], 1: [] });
  const sources: Sources = props.sources;

  useEffect(() => {
    inferenceWorker.warmupModels();
  });

  useEffect(() => {
    getMaybeMultipleChannelData(sources).then(sources => {
      let sourcesNoFile: Sources = [];
      for (let source of sources) {
        if (source && source.channelData) {
          sourcesNoFile.push({
            id: source.id,
            regionStarts: source.regionStarts,
            channelData: source.channelData,
          });
        }
      }
      const processed: Promise<Float32Array | undefined> = new Promise(
        async resolve => {
          if (
            sourcesNoFile[0] !== null &&
            sourcesNoFile[0]?.regionStarts[0] !== undefined
          ) {
            const resampledData = await inferenceWorker.downsampleSource(
              sourcesNoFile[0] as Source,
              0,
            );
            const embeddings = await inferenceWorker.runInference(
              resampledData,
              4,
            );
            console.log(embeddings);
            resolve(embeddings);
          }
        },
      );
    });
  }, [sources]);

  const handleSearchClick = ev => {
    const sources = props.sources;
    const startPoints = prepareRegionInfo(props.sources, regions);
  };

  return (
    <>
      <UploadSurferPair
        sources={props.sources}
        removeSource={props.removeSource}
        regions={regions}
        setRegions={setRegions}
        regionStartCallbacks={props.regionStartCallbacks}
      />
      <button onClick={handleSearchClick}>Analyze and Search</button>
    </>
  );
}

export default MusicSearchForm;
export type { Source, RegionStart };
