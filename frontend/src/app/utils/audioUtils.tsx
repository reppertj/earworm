type SampleRate = number;
type ChanneledAudioData = Float32Array[];
export type FullAudioData = [ChanneledAudioData, SampleRate];

declare global {
  interface Window {
    webkitAudioContext: typeof AudioContext;
  }
}

function makeRequest(method: string, url: string): Promise<ArrayBuffer> {
  return new Promise(function (resolve, reject) {
    var xhr = new XMLHttpRequest();
    xhr.open(method, url);
    xhr.responseType = 'arraybuffer';
    xhr.onload = function () {
      if (this.status >= 200 && this.status < 300) {
        const response: ArrayBuffer = xhr.response;
        resolve(response);
      } else {
        reject({
          status: this.status,
          statusText: xhr.statusText,
        });
      }
    };
    xhr.onerror = function () {
      reject({
        status: this.status,
        statusText: xhr.statusText,
      });
    };
    xhr.send();
  });
}

async function makeAudioBuffer(source: File | string): Promise<AudioBuffer> {
  const audioCtx = new (window.AudioContext || window.webkitAudioContext)();
  let audioData: Promise<ArrayBuffer>;
  if (typeof source === 'string') {
    audioData = makeRequest('GET', source);
  } else {
    audioData = source.arrayBuffer();
  }
  const audioBuffer = await audioCtx.decodeAudioData(await audioData);
  return audioBuffer;
}

export default async function getChannelDataAndSampleRate(
  source: File | string,
): Promise<FullAudioData> {
  const audioBuffer = await makeAudioBuffer(source);
  const offlineCtx = new OfflineAudioContext(
    audioBuffer.numberOfChannels,
    audioBuffer.duration * audioBuffer.sampleRate,
    audioBuffer.sampleRate,
  );
  if (audioBuffer.duration < 3) {
    throw new Error('AudioTooShort');
  }
  const offlineSource = offlineCtx.createBufferSource();
  offlineSource.buffer = audioBuffer;
  offlineSource.connect(offlineCtx.destination);
  offlineSource.start(0);
  const audioArray: Array<Float32Array> = [];
  await offlineCtx.startRendering().then(offlineBuffer => {
    for (var channel = 0; channel < offlineBuffer.numberOfChannels; channel++) {
      audioArray[channel] = offlineBuffer.getChannelData(channel);
    }
  });
  return [audioArray, audioBuffer.sampleRate];
}
