import { Tensor, InferenceSession } from "onnxjs";


const audio = "/Users/Justin/nc/likethisbutfree/modeling/src/data/audio_samples/1.mp3"
const session = new InferenceSession();
const url = "/Users/Justin/nc/likethisbutfree/frontend/src/inference/encoder_untrained.onnx";
session.loadModel(url);

function preprocess(data, width, height) {
    return null
};

const inputs = [
    new Tensor(new Float32Array([1.0, 2.0, 3.0, 4.0]), "float32", [2, 2]),
];



