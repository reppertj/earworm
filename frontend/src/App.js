import React, { useCallback, useEffect, useMemo } from "react";
import axios from "axios";
import "./App.css";
import { useDropzone } from "react-dropzone";
import { resample } from "wave-resampler";
import colormap from "colormap";
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
  const MIN_LENGTH = 3;
  if (buffer.duration < MIN_LENGTH + 1) {
    // Give ourselves a little breathing room
    throw new Error("AudioTooShort");
  }
  const TARGET_SAMPLE_RATE = 22050;
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
    "http://127.0.0.1:8081/conditionalInceptionEncoder/model.json";
  // TODO: Warmup the model separately to make the first prediction MUCH faster
  const firstModel = await tf.loadGraphModel(firstUrl);
  const FIRST_SHAPE = [1, 159500];
  // const warmup = await firstModel.executeAsync(
  //   { "input:0": tf.zeros(FIRST_SHAPE) },
  //   "Identity:0"
  // );
  // tf.dispose(warmup);
  const sgramTensor = await firstModel.executeAsync(
    { "input:0": tensor },
    "Identity:0"
  ); // Right now this is [128, 130]
  const sgram_reshaped = sgramTensor.expandDims().expandDims(); // [128, 624]
  const secondModel = await tf.loadGraphModel(secondUrl);
  const SECOND_SHAPE = [1, 1, 128, 129];
  // await secondModel.execute({"input_1:0": tf.zeros(SECOND_SHAPE)}, 'Identity:0')
  const latentTensor = secondModel.execute(
    { "input_1:0": sgram_reshaped },
    "Identity:0"
  );
  sgram_reshaped.dispose();
  console.log(latentTensor);
  return {sgramTensor, latentTensor};
}

async function tensorToImage(tensor) {
  const max = await tf.max(tensor);
  const min = await tf.min(tensor);
  await console.log(max.data())
  await console.log(min.data())
  const scaled = await tf.div(tf.sub(tensor, min), tf.sub(max, min));
  var values = await scaled.data();
  console.log(values)
  const N_SHADES = 72;
  values = Array.from(values);
  const colors = colormap({
    colormap: "jet",
    nshades: N_SHADES,
    format: "rba",
    alpha: 1,
  });
  console.log(colors)

  values = values.map((value) => {
    const index = Math.floor(value * N_SHADES);
    return [0, 0, 0, 1]
    // return colors[index];
  });
  var imageValues = new Uint8ClampedArray();
  imageValues = Uint8ClampedArray.from(values.flat());
  console.log("assigned as")
  console.log(imageValues)
  return imageValues;
}

function ShowImage(props) {
  const canvasRef = React.useRef();
  const draw = useCallback((ctx) => {
    var iData = ctx.getImageData(0, 0, props.width, props.height);
    console.log("here")
    console.log(props.buffer)
    iData.data.set(props.buffer);
    ctx.putImageData(iData, 0, 0);
  }, [props.buffer, props.height, props.width])


  useEffect(() => {
    const canvas = canvasRef.current
    const ctx = canvas.getContext("2d")
    if (props.buffer) {
      draw(ctx)
    }
  
    // if (props.buffer) {
    //   var iData = ctx.createImageData(props.width, props.height);
    //   console.log("Here")
    //   console.log(props.buffer)
    //   iData.rdata.set(props.buffer);
    //   ctx.putImageData(iData, 0, 0);
    // }  
  }, [draw, props.buffer])

  return (
    <div>
      <canvas ref={canvasRef} {...props} />
    </div>
  );
}

async function runInferenceOnFile(blob) {
  const audioCtx = new (window.AudioContext || window.webkitAudioContext)();
  const start = new Date();
  var buffer = await blob
    .arrayBuffer()
    .then((arrayBuffer) => audioCtx.decodeAudioData(arrayBuffer));
  tf.enableDebugMode(); // development
  // tf.enableProdMode(); // production TODO: Run inference in worker thread
  const { latentTensor, sgramTensor } = await resampleAudioBuffer(buffer)
    .then((resampled) => prepareTensor(resampled))
    .then((tensor) => runInference(tensor));
  const sgramData = await tensorToImage(sgramTensor);
  const end = new Date();
  const inferenceTime = end.getTime() - start.getTime();
  console.log(inferenceTime);
  return {
    audio: buffer,
    image: sgramData,
    latent: latentTensor,
    time: inferenceTime,
  };
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
  return <Dropzone />;
}

function SearchInfo(props) {
  return "Hello";
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

export { runInferenceOnFile, ShowImage };
