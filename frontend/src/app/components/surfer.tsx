import React, { useEffect, useRef, useState } from 'react';
import WaveSurfer from 'wavesurfer.js';
import RegionsPlugin from 'wavesurfer.js/dist/plugin/wavesurfer.regions';
import { Source, RegionStart } from './uploader';

const makeSurferOptions = (ref: HTMLDivElement) => ({
  container: ref,
  waveColor: '#eee',
  progressColor: 'OrangeRed',
  cursorColor: 'OrangeRed',
  barWidth: 2,
  barHeight: 1,
  responsive: true,
  scrollParent: true,
  height: 100,
  normalize: true,
  partialRender: true,
  plugins: [RegionsPlugin.create()],
});

function makeRegionOptions(id: string, start: number) {
  return {
    id: id,
    start: start,
    end: start + 3,
    loop: true,
    drag: true,
    resize: false,
    color: 'rgba(255, 99, 71, 0.5)',
  };
}

function UploadSurferPair(props) {
  const [freeRegionIDXs, setFreeRegionIDXs] = useState<Array<number>>([0, 1]);
  const [playing, setPlaying] = useState([false, false]);
  const [first, second]: Source[] = props.sources;

  const regions = props.regions;
  const setRegions = props.setRegions;
  const {
    addRegionStart,
    removeRegionStart,
    updateRegionStart,
  } = props.regionStartCallbacks;

  const onPlayPause = (isPlaying: boolean, index: number) => {
    const other = index === 0 ? 1 : 0;
    const nowPlaying = playing.slice();
    nowPlaying[index] = isPlaying;
    if (isPlaying) {
      nowPlaying[other] = !isPlaying;
    }
    setPlaying(nowPlaying);
  };

  const handleAddRegion = (sourceID, surferID, waveSurfer: WaveSurfer) => {
    if (freeRegionIDXs.length) {
      const newFreeRegionIDXs = freeRegionIDXs.slice();
      const regionIDX = newFreeRegionIDXs[0];
      newFreeRegionIDXs.shift();
      setFreeRegionIDXs(newFreeRegionIDXs);
      const startPosition = Math.min(
        waveSurfer.getCurrentTime(),
        waveSurfer.getDuration() - 3,
      );
      const region = waveSurfer.addRegion(
        makeRegionOptions((regionIDX + 1).toString(), startPosition),
      );
      const regionID = addRegionStart(sourceID, startPosition);
      region.regionID = regionID;
      regions[surferID] = regions[surferID].concat(region);
      region.on('update-end', function () {
        updateRegionStart(sourceID, regionID, region.start);
      });
      setRegions(regions);
      return region;
    }
  };

  const handleRemoveRegion = (sourceID, surferID, region): void => {
    const regionIDX = parseInt(region.id) - 1;
    regionIDX === 0
      ? freeRegionIDXs.unshift(regionIDX)
      : freeRegionIDXs.push(regionIDX);
    setFreeRegionIDXs(freeRegionIDXs);
    region.remove();
    regions[surferID] = regions[surferID].filter(reg => {
      return reg.id !== region.id;
    });
    setRegions(regions);
    removeRegionStart(sourceID, region.regionID);
  };

  return (
    <>
      <UploadSurfer
        surferID={0}
        source={first}
        removeSource={() => props.removeSource(0)}
        handleAddRegion={handleAddRegion}
        handleRemoveRegion={handleRemoveRegion}
        freeRegions={freeRegionIDXs.length}
        regions={regions[0]}
        playing={playing[0]}
        onPlayPause={(isPlaying: boolean) => onPlayPause(isPlaying, 0)}
      />
      <UploadSurfer
        surferID={1}
        source={second}
        removeSource={() => props.removeSource(1)}
        handleAddRegion={handleAddRegion}
        handleRemoveRegion={handleRemoveRegion}
        freeRegions={freeRegionIDXs.length}
        regions={regions[1]}
        playing={playing[1]}
        onPlayPause={(isPlaying: boolean) => onPlayPause(isPlaying, 1)}
      />
    </>
  );
}

function UploadSurfer(props) {
  const waveformRef = useRef<HTMLDivElement | null>(null);
  const wavesurfer = useRef<WaveSurfer | null>(null);
  const [volume, setVolume] = useState(0.5);
  const [regions, setRegions] = useState(props.regions);
  const [loaded, setLoaded] = useState(false);

  const source =
    props.source?.source === undefined ? null : props.source.source;
  const surferID = props.surferID;

  // Create new instance on mount and when source changes
  useEffect(() => {
    if (waveformRef.current) {
      props.onPlayPause(false, surferID);
      const options = makeSurferOptions(waveformRef.current);
      wavesurfer.current = WaveSurfer.create(options);
      if (source === null) {
      } else if (typeof source === 'string') {
        wavesurfer.current.load(source);
      } else {
        wavesurfer.current.loadBlob(source);
      }
    }
    return () => {
      if (wavesurfer.current) {
        wavesurfer.current?.destroy();
        setLoaded(false);
        setRegions(props.regions);
      }
    };
  }, [source, surferID]);

  useEffect(() => {
    props.playing ? wavesurfer.current?.play() : wavesurfer.current?.pause();
  }, [props.playing]);

  // Set volume in case of new instance with volume already set
  wavesurfer.current?.on('ready', function () {
    if (wavesurfer.current) {
      setLoaded(true);
      wavesurfer.current.setVolume(volume);
      setVolume(volume);
    }
  });

  function addRegion() {
    if (wavesurfer.current && loaded) {
      const region = props.handleAddRegion(
        props.source.id,
        props.surferID,
        wavesurfer.current,
      );
      setRegions(regions.concat(region));
    }
  }

  function removeRegion(region) {
    if (wavesurfer.current) {
      setRegions(
        regions.filter(reg => {
          return reg.id !== region.id;
        }),
      );
      props.handleRemoveRegion(props.source.id, props.surferID, region);
    }
  }

  function handleRemoveSource() {
    for (let region of regions) {
      removeRegion(region);
    }
    props.removeSource();
  }

  // useEffect(() => {
  //   if (waveformRef.current) {
  //     props.onRegionUpdate()
  //   }
  // }, [props.n_regions]);

  const handlePlayPause = () => {
    if (wavesurfer.current) {
      wavesurfer.current.playPause();
      const nowPlaying = wavesurfer.current.isPlaying();
      props.onPlayPause(nowPlaying);
    }
  };

  function onVolumeChange(ev) {
    const { target } = ev;
    const newVolume = +target.value;

    if (newVolume) {
      setVolume(newVolume);
      wavesurfer.current?.setVolume(newVolume || 1);
    }
  }

  return (
    <div>
      <div id="waveform" ref={waveformRef} />
      <div className="controls">
        {loaded ? <button onClick={handleRemoveSource}>Clear</button> : null}
        <button onClick={handlePlayPause} disabled={!loaded}>
          {!props.playing ? 'Play' : 'Pause'}
        </button>
        <button onClick={addRegion} disabled={!loaded || !props.freeRegions}>
          Add Focus Zone ({props.freeRegions})
        </button>
        {regions.map(region => {
          if (region) {
            return (
              <button key={region.id} onClick={ev => removeRegion(region)}>
                Remove Zone {region.id}
              </button>
            );
          } else {
            return null;
          }
        })}
        <input
          type="range"
          id="volume"
          name="volume"
          min="0.01" // 0 == 1 for wavesurfer, so use a small value for min
          max="1"
          step=".025"
          onChange={onVolumeChange}
          defaultValue={volume}
        />
        <label htmlFor="volume">Volume</label>
      </div>
    </div>
  );
}

function ResultSurfer(props) {
  return <></>;
}

export { UploadSurferPair, ResultSurfer };
