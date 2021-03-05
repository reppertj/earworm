import React, { useCallback, useEffect } from "react";
import axios from "axios";
import "./App.css";
import { useDropzone } from "react-dropzone";
import { resample } from "wave-resampler";
import ndarray from "ndarray";
import colormap from "colormap";
import ops from "ndarray-ops";
import * as tf from "@tensorflow/tfjs";

function getRandomInt(max) {
  max = Math.floor(max);
  return Math.floor(Math.random() * max);
}

async function resampleAudioBuffer(buffer) {
  const resampledChannelData = new Array(buffer.numberOfChannels);
  const offlineCtx = new OfflineAudioContext(
    buffer.numberOfChannels,
    buffer.duration * buffer.sampleRate,
    buffer.sampleRate
  );
  const MIN_LENGTH = 10;
  if (buffer.duration < MIN_LENGTH + 1) {
    // Give ourselves a little breathing room
    throw new Error("AudioTooShort");
  }
  const TARGET_SAMPLE_RATE = 15950;
  const TARGET_SAMPLES = TARGET_SAMPLE_RATE * MIN_LENGTH;
  const start_idx = getRandomInt(
    TARGET_SAMPLE_RATE * buffer.duration - TARGET_SAMPLES
  );
  const end_idx = start_idx + TARGET_SAMPLES;
  const offlineSource = offlineCtx.createBufferSource();
  offlineSource.buffer = buffer;
  offlineSource.connect(offlineCtx.destination);
  offlineSource.start(0);
  await offlineCtx.startRendering().then((buffer) => {
    for (var channel = 0; channel < buffer.numberOfChannels; channel++) {
      var audioArray = buffer.getChannelData(channel);
      var resampled = resample(
        audioArray,
        buffer.sampleRate,
        TARGET_SAMPLE_RATE,
        { method: "sinc" }
      );
      resampled = Float32Array.from(resampled);
      resampledChannelData[channel] = resampled.slice(start_idx, end_idx);
    }
  });
  return resampledChannelData;
}

async function prepareTensor(channelArray) {
  const inputShape = [channelArray.length, channelArray[0].length];
  const tensorNd = tf.tensor(channelArray, inputShape);
  const tensor = tf.mean(tensorNd, 0, true);
  tf.dispose(tensorNd);
  return tensor;
}

async function runInference(tensor) {
  // Intended for local dev use only
  const firstUrl = "http://127.0.0.1:8081/waveformSpectrogramModel/model.json";
  const secondUrl =
    "http://127.0.0.1:8081/SpectrogramEmbeddingModel/model.json";
  // TODO: Warmup the model separately to make the first prediction faster
  const firstModel = await tf.loadGraphModel(firstUrl);
  const FIRST_SHAPE = [1, 159500];
  const warmup = await firstModel.executeAsync(
    { "input:0": tf.zeros(FIRST_SHAPE) },
    "Identity:0"
  );
  tf.dispose(warmup);
  const sgram = await firstModel.executeAsync(
    { "input:0": tensor },
    "Identity:0"
  );
  // const secondModel = await tf.loadGraphModel(secondUrl)
  // const SECOND_SHAPE = [1, 1, 128, 624]
  // await secondModel.execute({"input_1:0": tf.zeros(SECOND_SHAPE)}, 'Identity:0')
  return sgram;
}

async function tensorToImage(tensor) {
  const max = await tf.max(tensor);
  const min = await tf.min(tensor);
  const scaled = await tf.div(tf.sub(tensor, min), tf.sub(max, min));
  var values = await tf.transpose(scaled).data();
  const N_SHADES = 72;
  values = Array.from(values);
  const colors = colormap({
    colormap: "jet",
    nshades: N_SHADES,
    format: "rba",
    alpha: 1,
  });
  values = values.map((value) => {
    const index = Math.floor(value * N_SHADES);
    return colors[index];
  });
  let imageData = new Uint8Array();
  imageData = Uint8Array.from(values.flat());
  console.log(imageData);
}


function ShowImage(props) {
  const imageData = props.imageData
  const canvasRef = React.useRef(null)
  return (
    <div>
      <canvas ref={canvasRef} width={624} height={128} />
    </div>
  );
}

async function runInferenceOnFile(blob) {
  const audioCtx = new (window.AudioContext || window.webkitAudioContext)();
  const start = new Date();
  var buffer = await blob
    .arrayBuffer()
    .then((arrayBuffer) => audioCtx.decodeAudioData(arrayBuffer));
  tf.enableDebugMode(); // remove in production
  const latentVector = await resampleAudioBuffer(buffer)
    .then((resampled) => prepareTensor(resampled))
    .then((tensor) => runInference(tensor));
  await tensorToImage(latentVector);
  const end = new Date();
  const inferenceTime = end.getTime() - start.getTime();
  console.log(latentVector);
  console.log(inferenceTime);
  return [buffer, latentVector, inferenceTime];
}

function Dropzone(props) {
  const onDrop = useCallback(
    (acceptedFiles) => runInferenceOnFile(acceptedFiles[0]),
    []
  );

  const {
    acceptedFiles,
    fileRejections,
    getRootProps,
    getInputProps,
  } = useDropzone({
    accept: "audio/*",
    onDrop,
    maxFiles: 1,
  });

  const acceptedFileItems = acceptedFiles.map((file) => (
    <li key={file.path}>
      {file.path} - {file.size} bytes
    </li>
  ));

  const fileRejectionItems = fileRejections.map(({ file, errors }) => (
    <li key={file.path}>
      {file.path} - {file.size} bytes
      <ul>
        {errors.map((e) => (
          <li key={e.code}>{e.message}</li>
        ))}
      </ul>
    </li>
  ));

  return (
    <section className="container">
      <div {...getRootProps({ className: "dropzone" })}>
        <input {...getInputProps()} />
        <p>Drag and drop an audio file here, or click to select a file.</p>
      </div>
      <aside>
        <h4>Accepted files</h4>
        <ul>{acceptedFileItems}</ul>
        <h4>Rejected files</h4>
        <ul>{fileRejectionItems}</ul>
      </aside>
    </section>
  );
}

function SearchBar(props) {
  return (
    <Dropzone />
  );
}

function SearchInfo(props) {
  return (
    "Hello"
  );
}

function NeighborSearch(props) {
  return (
    <div>
      <SearchBar />
      <SearchInfo />
    </div>
  );
}

function App(props) {
  return (
    <div className="App">
      <header className="App-header">
        <NeighborSearch />
      </header>
    </div>
  );
}

export { runInferenceOnFile }