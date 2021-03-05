import { useCallback } from "react";
import { useDropzone } from "react-dropzone";

import { runInferenceOnFile } from "../App"

const dropStyle = {};
const dropActiveStyle = {};

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
      isDragActive,
      isDragAccept,
      isDragReject
    } = useDropzone({
      accept: "audio/*",
      onDrop,
      style: dropStyle,
      activeStyle: dropActiveStyle,
      className: "",
      disableClick: true,
      maxFiles: 1,
    });
  
    const acceptedFileItems = acceptedFiles.map((file) => (
      <li key={file.path}>
        {file.path} - {Math.round(((file.size / 1e6) + Number.EPSILON) * 10) / 10} MB
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
      <div>
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
      </div>
    );
  }

  export default Dropzone;