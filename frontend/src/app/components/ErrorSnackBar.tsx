import React from 'react';
import Snackbar from '@material-ui/core/Snackbar';
import MuiAlert, { AlertProps } from '@material-ui/lab/Alert';

function Alert(props: AlertProps) {
  return <MuiAlert elevation={6} variant="filled" {...props} />;
}

interface Props {
  hasError: {
    error: boolean;
    message: string;
  };
  setHasError: React.Dispatch<
    React.SetStateAction<{
      error: boolean;
      message: string;
    }>
  >;
}

export default function ErrorSnackBar(props: Props) {
  const resetErrorState = () => {
    props.setHasError({ error: false, message: '' });
  };

  return (
    <>
      {props.hasError.error && (
        <Snackbar open={true} autoHideDuration={8000} onClose={resetErrorState}>
          <Alert severity="error">{props.hasError.message}</Alert>
        </Snackbar>
      )}
    </>
  );
}
