import React, { useCallback } from "react";
import axios from "axios";
import logo from './logo.svg';
import './App.css';
import {useDropzone} from 'react-dropzone';
import { resample } from 'wave-resampler';
import { Tensor, InferenceSession } from "onnxjs";
import ndarray from 'ndarray';
import ops from 'ndarray-ops';
import r from 'ndarray-concat-rows';
import prep from './preprocess_model.onnx';

function getRandomInt(min, max) {
  min = Math.ceil(min);
  max = Math.floor(max);
  return Math.floor(Math.random() * (max - min) + min);
}

async function preprocess(list_of_ndarrays) {
  // TODO: Also normalize output before model

  const session = new InferenceSession(); // Initialize early to warmup?
  const url = prep;
  session.loadModel(url);

}

function Dropzone(props) {
  const onDrop = useCallback((acceptedFiles) => {
    acceptedFiles.forEach((file) => {
      const reader = new FileReader()
      reader.onabort = () => console.log('file reader was aborted')
      reader.onerror = () => console.log('file reading failed')
      reader.onload = (e) => {
        var audioCtx = new (window.AudioContext || window.webkitAudioContext)();
        audioCtx.decodeAudioData(e.target.result).then(function(buffer) {
          var soundSource = audioCtx.createBufferSource();
          soundSource.buffer = buffer;
          const sourceSampleRate = buffer.sampleRate;
          const sourceDuration = buffer.duration;
          console.log(sourceDuration)
          // TODO: Reject too short
          soundSource.connect(audioCtx.destination);
          soundSource.start(0);
          console.log(sourceSampleRate)
          const TARGET_SAMPLE_RATE = 15950;
          const offlineCtx = new OfflineAudioContext(buffer.numberOfChannels, buffer.duration * buffer.sampleRate, buffer.sampleRate);
          const cloneBuffer = offlineCtx.createBuffer(buffer.numberOfChannels, buffer.length, buffer.sampleRate);
          for (let channel = 0; channel < buffer.numberOfChannels; channel++) {
            cloneBuffer.copyToChannel(buffer.getChannelData(channel), channel);
          }
          var offlineSource = offlineCtx.createBufferSource();
          offlineSource.buffer = cloneBuffer;
          offlineSource.connect(offlineCtx.destination);
          offlineCtx.oncomplete = function(e) {
            const resampledAudioBuffer = e.renderedBuffer;
            console.log(resampledAudioBuffer.getChannelData(0).length)
            var resampledAudioArrays = []
            for (let channel = 0; channel < resampledAudioBuffer.numberOfChannels; channel++) {
              let audioArray = resampledAudioBuffer.getChannelData(channel);
              let resampled = resample(audioArray, sourceSampleRate, TARGET_SAMPLE_RATE, {method: "sinc"});
              console.log(resampled);
              var resampled_ndarray = ndarray(resampled, [1, resampled.length])
              resampledAudioArrays.push(resampled_ndarray);
            }
            var ndarray_waveform = r(resampledAudioArrays)
            preprocess(ndarray_waveform)
          }
          offlineCtx.startRendering();
          offlineSource.start(0);
        });
      }
      reader.readAsArrayBuffer(file);
  })
}, [])

const {
  acceptedFiles,
  fileRejections,
  getRootProps,
  getInputProps
} = useDropzone({
  accept: 'audio/*',
  onDrop,
  maxFiles:1
});

  const acceptedFileItems = acceptedFiles.map(file => (
    <li key={file.path}>
      {file.path} - {file.size} bytes
    </li>
  ));

  const fileRejectionItems = fileRejections.map(({ file, errors }) => (
    <li key={file.path}>
      {file.path} - {file.size} bytes
      <ul>
        {errors.map(e => (
          <li key={e.code}>{e.message}</li>
        ))}
      </ul>
    </li>
  ));

  return (
    <section className="container">
      <div {...getRootProps({ className: 'dropzone' })}>
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

function App (props) {
    return (
      <div className="App">
        <header className="App-header">
          <Dropzone />
        </header>
      </div>
    );  
 
}

export default App;
