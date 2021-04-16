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

export const ErrorSnackBar = React.forwardRef((props: Props, ref) => {
  const resetErrorState = () => {
    props.setHasError({ error: false, message: '' });
  };

  return (
    <div>
      {props.hasError.error && (
        <Snackbar
          ref={ref}
          open={true}
          autoHideDuration={8000}
          onClose={resetErrorState}
        >
          <Alert severity="error">{props.hasError.message}</Alert>
        </Snackbar>
      )}
    </div>
  );
});
