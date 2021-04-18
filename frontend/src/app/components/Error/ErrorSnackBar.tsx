import React from 'react';
import Snackbar from '@material-ui/core/Snackbar';
import MuiAlert, { AlertProps } from '@material-ui/lab/Alert';
import { useSelector, useDispatch } from 'react-redux';
import { useErrorSlice } from './slice';
import { selectAllErrors } from './slice/selectors';

function Alert(props: AlertProps) {
  return <MuiAlert elevation={6} variant="filled" {...props} />;
}

export const ErrorSnackBar = React.forwardRef((props, ref) => {
  const errors = useSelector(selectAllErrors);
  const dispatch = useDispatch();
  const { actions } = useErrorSlice();

  const clearOneError = (errorId: string) => {
    dispatch(actions.errorRemoved(errorId));
  };

  const bottomError = errors[0];

  return (
    <div>
      {bottomError && (
        <Snackbar
          ref={ref}
          open={true}
          autoHideDuration={8000}
          onClose={() => clearOneError(bottomError.id)}
        >
          <Alert severity="error">{bottomError.message}</Alert>
        </Snackbar>
      )}
    </div>
  );
});
