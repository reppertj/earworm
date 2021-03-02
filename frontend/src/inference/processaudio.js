function async

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