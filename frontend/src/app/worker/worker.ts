import * as tf from '@tensorflow/tfjs';
import { resample } from 'wave-resampler';
import { Source } from 'app/components/SearchForm';

var spectrogramModel: tf.GraphModel;
var encodingModel: tf.GraphModel;
var embeddingModel: tf.GraphModel;
var downloadError: boolean = false;
var warmedUp: boolean = false;

// tf.enableDebugMode();
tf.enableProdMode();

async function downloadModels() {
  try {
    const spectrogramModelUrl =
      'http://127.0.0.1:8081/MAKE_SPECTROGRAM/model.json';
    const encodingModelUrl = 'http://127.0.0.1:8081/MAKE_ENCODING/model.json';
    const embeddingModelUrl = 'http://127.0.0.1:8081/MAKE_EMBEDDING/model.json';
    spectrogramModel = await tf.loadGraphModel(spectrogramModelUrl);
    encodingModel = await tf.loadGraphModel(encodingModelUrl);
    embeddingModel = await tf.loadGraphModel(embeddingModelUrl);
    return true;
  } catch {
    downloadError = true;
  }
}

type downloadStatus = 'download error' | 'not downloaded' | 'downloaded';

export function getDownloadStatus(): downloadStatus {
  if (downloadError) {
    return 'download error';
  } else if (
    [spectrogramModel, encodingModel, embeddingModel].some(
      mod => mod === undefined,
    )
  ) {
    return 'not downloaded';
  } else {
    return 'downloaded';
  }
}

export async function warmupModels() {
  if (!warmedUp) {
    if (getDownloadStatus() !== 'downloaded') {
      await downloadModels();
    }
    try {
      const WAVEFORM_SHAPE = [1, 22050 * 3];
      const warmed_sgram = (await spectrogramModel.executeAsync(
        { 'input_1:0': tf.zeros(WAVEFORM_SHAPE) },
        'Identity:0',
      )) as tf.Tensor;
      const warmed_encoding = encodingModel.execute(
        { 'unknown:0': warmed_sgram },
        'Identity:0',
      ) as tf.Tensor;
      tf.dispose(warmed_sgram);
      const warmed_embedding = embeddingModel.execute(
        { 'unknown:0': warmed_encoding, unknown_0: tf.zeros([], 'int32') },
        'Identity: 0',
      );
      tf.dispose(warmed_encoding);
      tf.dispose(warmed_embedding);
      warmedUp = true;
      return true;
    } catch {
      console.log('Error warming up models');
    }
  } else {
    return true;
  }
}

async function prepareTensor(channelArray: Float32Array[]): Promise<tf.Tensor> {
  const inputShape = [channelArray.length, channelArray[0].length];
  const tensorNd = tf.tensor(channelArray, inputShape);
  const tensor = tf.mean(tensorNd, 0, true);
  tf.dispose(tensorNd);
  return tensor;
}

export async function runInference(
  channelArray: Float32Array[],
  preference: number,
) {
  try {
    if (getDownloadStatus() !== 'downloaded') {
      await downloadModels();
    }
    console.log('Inference using backend:', tf.getBackend());
    const inputTensor = await prepareTensor(channelArray);
    const sgram = (await spectrogramModel.executeAsync(
      { 'input_1:0': inputTensor },
      'Identity:0',
    )) as tf.Tensor;
    const encoded = encodingModel.execute(
      { 'unknown:0': sgram },
      'Identity:0',
    ) as tf.Tensor;
    tf.dispose(sgram);
    const embedded = embeddingModel.execute(
      { 'unknown:0': encoded, unknown_0: tf.scalar(preference, 'int32') },
      'Identity: 0',
    ) as tf.Tensor;
    tf.dispose(encoded);
    const embeddingsData = await embedded.data();
    tf.dispose(embedded);
    warmedUp = true;
    return embeddingsData;
  } catch {
    console.log('Error running inference');
  }
}

export function downsampleSource(source: Source, regionStartIndex: number) {
  const { channelData, regionStarts } = source;
  const regionStartSeconds = regionStarts[regionStartIndex].seconds;
  const resampledAudioData: Float32Array[] = [];
  if (channelData) {
    const [channeledAudioData, sampleRate] = channelData;
    for (let channel of channeledAudioData) {
      const sliced = channel.slice(
        sampleRate * regionStartSeconds,
        sampleRate * (regionStartSeconds + 3),
      );
      const resampled = resample(sliced, sampleRate, 22050, {
        method: 'sinc',
      });
      const resampledAsF32 = Float32Array.from(resampled);
      resampledAudioData.push(resampledAsF32);
    }
  }
  return resampledAudioData;
}
