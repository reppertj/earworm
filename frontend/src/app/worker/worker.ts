import { Source } from '../components/SearchInputs/AudioInputs/slice/types';
import * as tf from '@tensorflow/tfjs';
import { resample } from 'wave-resampler';

var spectrogramModel: tf.GraphModel;
var encodingModel: tf.GraphModel;
var embeddingModel: tf.GraphModel;
var downloadError: boolean = false;
var warmedUp: boolean = false;

// tf.enableDebugMode();
tf.enableProdMode();

async function downloadModels() {
  try {
    const spectrogramModelUrl = '/embeddings/MAKE_SPECTROGRAM/model.json';
    const encodingModelUrl = '/embeddings/MAKE_ENCODING/model.json';
    const embeddingModelUrl = '/embeddings/MAKE_EMBEDDING/model.json';
    spectrogramModel = await tf.loadGraphModel(spectrogramModelUrl);
    encodingModel = await tf.loadGraphModel(encodingModelUrl);
    embeddingModel = await tf.loadGraphModel(embeddingModelUrl);
    return true;
  } catch (err) {
    console.log('Error downloading models:', err);
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
      const start = Date.now();
      const WAVEFORM_SHAPE = [1, 22050 * 3];
      const modelInput = tf.zeros(WAVEFORM_SHAPE);
      const warmedSgram = (await spectrogramModel.executeAsync(
        { 'unknown:0': modelInput },
        'Identity:0',
      )) as tf.Tensor;
      tf.dispose(modelInput);
      const warmedEncoding = encodingModel.execute(
        { 'unknown:0': warmedSgram },
        'Identity:0',
      ) as tf.Tensor;
      tf.dispose(warmedSgram);
      const samplePreference = tf.zeros([], 'int32');
      const warmed_embedding = embeddingModel.execute(
        { 'unknown:0': warmedEncoding, unknown_0: samplePreference },
        'Identity: 0',
      );
      tf.dispose(samplePreference);
      tf.dispose(warmedEncoding);
      tf.dispose(warmed_embedding);
      warmedUp = true;
      const end = Date.now();
      console.log(
        `Warmed up models using ${tf.getBackend()} backend in ${
          end - start
        } ms`,
      );
    } catch (err) {
      console.log('Error warming up models:', err);
    }
  }
}

async function prepareTensor(channelArray: Float32Array[]): Promise<tf.Tensor> {
  const inputShape = [channelArray.length, channelArray[0].length];
  const tensorNd = tf.tensor(channelArray, inputShape);
  const tensor2d = tf.mean(tensorNd, 0, true);
  tf.dispose(tensorNd); // Manual memory management using webgl backend

  // Because of rounding, the waveform length may be 1 sample shy
  // interestingly, this only seems to happen on Chrome
  // We'll just zero-pad it

  if (inputShape[1] === 22049) {
    const pad = tf.zeros([1, 1], 'float32');
    const padded = tf.concat([tensor2d, pad], 1);
    tf.dispose(tensor2d);
    tf.dispose(pad);
    return padded;
  } else {
    return tensor2d;
  }
}

/**
 * Prepares tfjs tensors from waveform data, then
 * runs full inference on all three models:
 * 1. Spectrogram - DB-scaled MEL spectrogram
 * 2. Encoding - CNN based on MobileNetV2
 * 3. Embedding - MLP that takes encoding output and "word" embedding
 * with the preference scalar as inputs and produces an embedding
 * @param channelArray Array of waveform data (one per channel)
 * @param preference Single integer input to the embedding model
 * @returns
 */
export async function runInference(
  channelArray: Float32Array[],
  preference: number,
) {
  try {
    if (getDownloadStatus() !== 'downloaded') {
      await downloadModels();
    }
    const start = Date.now();
    const inputTensor = await prepareTensor(channelArray);
    const sgram = (await spectrogramModel.executeAsync(
      { 'unknown:0': inputTensor },
      'Identity:0',
    )) as tf.Tensor;
    tf.dispose(inputTensor);
    const encoded = encodingModel.execute(
      { 'unknown:0': sgram },
      'Identity:0',
    ) as tf.Tensor;
    tf.dispose(sgram);
    const preferenceTensor = tf.scalar(preference, 'int32');
    const embedded = embeddingModel.execute(
      { 'unknown:0': encoded, unknown_0: preferenceTensor },
      'Identity: 0',
    ) as tf.Tensor;
    tf.dispose(preferenceTensor);
    tf.dispose(encoded);
    const embeddingsData = await embedded.data();
    tf.dispose(embedded);
    warmedUp = true;
    const end = Date.now();
    console.log(
      `Ran inference using ${tf.getBackend()} backend in ${end - start} ms`,
    );
    return Array.from(embeddingsData);
  } catch (err) {
    console.log('Error running inference:', err);
    throw err;
  }
}

export function downsampleSource(source: Source, startTime: number) {
  try {
    const { channelData, sampleRate } = source;
    const resampledAudioData: Float32Array[] = [];
    if (channelData && sampleRate) {
      for (let channel of channelData) {
        const start = Math.min(
          sampleRate * startTime,
          channel.length - 3.01 * sampleRate,
        );
        const end = start + sampleRate * 3;
        const sliced = channel.slice(start, end);
        const resampled = resample(sliced, sampleRate, 22050, {
          method: 'sinc',
        });
        const resampledAsF32 = Float32Array.from(resampled);
        resampledAudioData.push(resampledAsF32);
      }
      return resampledAudioData;
    }
  } catch (err) {
    console.log('Error downsampling', err);
  }
}
