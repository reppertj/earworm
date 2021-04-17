import { AudioContext, OfflineAudioContext } from 'standardized-audio-context';

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

/**
 * Get an audio buffer from a file or url
 * @param source    Blob or url
 */
async function makeAudioBuffer(source: File | string): Promise<AudioBuffer> {
  const audioCtx = new AudioContext();
  let audioData: Promise<ArrayBuffer>;
  if (typeof source === 'string') {
    audioData = makeRequest('GET', source);
  } else {
    audioData = source.arrayBuffer();
  }
  const audioBuffer = await audioCtx.decodeAudioData(await audioData);
  return audioBuffer;
}

/**
 * Retrieve array containing typed arrys of waveform data
 * (one per channel in the source)
 * and the sample rate as determined by the webaudio API
 * @param source    blob or url
 * @returns
 */
export default async function getChannelDataAndSampleRate(
  source: File | string,
): Promise<{ channelData: Readonly<Float32Array[]>; sampleRate: number }> {
  const audioBuffer = await makeAudioBuffer(source);
  const offlineCtx = new OfflineAudioContext(
    audioBuffer.numberOfChannels,
    audioBuffer.duration * audioBuffer.sampleRate,
    audioBuffer.sampleRate,
  );
  if (audioBuffer.duration < 4) {
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
  const channelData = Object.freeze(audioArray);
  const sampleRate = audioBuffer.sampleRate;
  return { channelData, sampleRate };
}
