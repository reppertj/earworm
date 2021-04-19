/**
 *
 * LicenseModal
 *
 */
import * as React from 'react';
import { useSelector } from 'react-redux';
import { selectResultsExist } from '../slice/selectors';
import Dialog from '@material-ui/core/Dialog';
import DialogActions from '@material-ui/core/DialogActions';
import DialogContent from '@material-ui/core/DialogContent';
import DialogContentText from '@material-ui/core/DialogContentText';
import DialogTitle from '@material-ui/core/DialogTitle';
import Slide from '@material-ui/core/Slide';
import Typography from '@material-ui/core/Typography';
import Button from '@material-ui/core/Button';
import { TransitionProps } from '@material-ui/core/transitions';
import DoneIcon from '@material-ui/icons/Done';

const Transition = React.forwardRef(function Transition(
  props: TransitionProps & { children?: React.ReactElement<any, any> },
  ref: React.Ref<unknown>,
) {
  return <Slide direction="up" ref={ref} {...props} />;
});

interface Props {}

export function LicenseModal(props: Props) {
  const [viewedOnce, setViewedOnce] = React.useState(false);

  const resultsExist = useSelector(selectResultsExist);

  const handleClose = () => {
    setViewedOnce(true);
  };
  return (
    <div>
      <Dialog
        open={!viewedOnce && resultsExist}
        onClose={handleClose}
        TransitionComponent={Transition}
        keepMounted
        onKeyUp={e => {
          if (e.key === 'Enter') {
            handleClose();
          }
        }}
      >
        <DialogTitle id="license-alart-dialog-title">
          {'Most licenses require attribution!'}
        </DialogTitle>
        <DialogContent>
          <DialogContentText
            component={'span'}
            id="license-alert-dialog-description"
          >
            <Typography component={'span'} variant="body1">
              <ol>
                <li>
                  Many licenses restrict certain uses,{' '}
                  <strong>including some commercial uses.</strong>
                </li>
                <li>
                  License information provided here{' '}
                  <strong>may be incorrect</strong> or out-of-date.
                </li>
                <li>
                  Always <strong>verify the license terms</strong> before using
                  a song in a project; if in doubt, contact the creator or
                  service providing the song.
                </li>
                <li>
                  <strong>Earworm does not host downloads</strong>, only links
                  to other services.
                </li>
                <li>
                  Many artists accept donations—
                  <strong>consider donating</strong> to artists when you use
                  their music!
                </li>
              </ol>
            </Typography>
          </DialogContentText>
        </DialogContent>
        <DialogActions>
          <Button
            onClick={handleClose}
            variant="contained"
            color="primary"
            autoFocus
            startIcon={<DoneIcon />}
          >
            I understand
          </Button>
        </DialogActions>
      </Dialog>
    </div>
  );
}
