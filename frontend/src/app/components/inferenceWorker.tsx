import { Source } from './uploader';

onmessage = ev => {
  const { audioData } = ev.data;
  console.log(audioData);
};

// async function prepareChannelData(source: Source) {

// }

// async function runInference(source: Source, startSeconds: number) {
//   const start = new Date();
//   const channelData = makeChannelData(source);
//   tf.enableDebugMode(); // development
//   // tf.enableProdMode(); // production TODO: Run inference in worker thread
//   const { latentTensor, sgramTensor } = prepareTensor(channelData))
//     .then(tensor => runInference(tensor));
//   const sgramData = await tensorToImage(sgramTensor);
//   const end = new Date();
//   const inferenceTime = end.getTime() - start.getTime();
//   console.log(inferenceTime);
//   return {
//     audio: buffer,
//     image: sgramData,
//     latent: latentTensor,
//     time: inferenceTime,
//   };
// }
