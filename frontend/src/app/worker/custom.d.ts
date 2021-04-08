interface RegionStart {
  id: number;
  seconds: number;
  preference: number;
}

interface Source {
  id: number;
  regionStarts: RegionStart[];
  source?: string | File;
  channelData?: FullAudioData;
}

declare module 'comlink-loader!*' {
  class WebpackWorker extends Worker {
    constructor();

    // Add declarations here; return type should be wrapped in a promise

    warmupModels(): Promise<boolean>;
    downsampleSource(
      source: Source,
      regionStartIdx: number,
    ): Promise<Float32Array[]>;
    runInference(
      channelArray: Float32Array[],
      preference: number,
    ): Promise<Float32Array>;
  }
  export = WebpackWorker;
}
