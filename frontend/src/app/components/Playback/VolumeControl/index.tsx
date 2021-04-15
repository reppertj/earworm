/**
 *
 * VolumeControl
 *
 */
import React, { memo, useState } from 'react';
import { useSelector, useDispatch } from 'react-redux';
import { makeStyles } from '@material-ui/core/styles';
import Slider from '@material-ui/core/Slider';
import Grid from '@material-ui/core/Grid';
import VolumeDown from '@material-ui/icons/VolumeDown';
import VolumeUp from '@material-ui/icons/VolumeUp';
import { useVolumeSlice } from '../volumeSlice';
import { selectVolume } from '../volumeSlice/selectors';

const useStyles = makeStyles({
  root: {
    width: 175,
  },
});

export const VolumeControl = memo(() => {
  const classes = useStyles();
  const dispatch = useDispatch();
  const volume = useSelector(selectVolume);
  const { actions } = useVolumeSlice();
  const [value, setValue] = useState(volume.value);
  const handleChange = (e: any, newValue: number | number[]) => {
    dispatch(actions.changeVolume(newValue as number));
    setValue(newValue as number);
  };

  return (
    <div className={classes.root}>
      <Grid container spacing={2}>
        <Grid item>
          <VolumeDown fontSize="small" />
        </Grid>
        <Grid item xs>
          <Slider
            value={value}
            step={0.01}
            min={0.001}
            max={1}
            color="secondary"
            onChange={handleChange}
            aria-labelledby="continuous-slider"
          />
        </Grid>
        <Grid item>
          <VolumeUp fontSize="small" />
        </Grid>
      </Grid>
    </div>
    // <div className={classes.root}>
    //   <Slider
    //     onChange={handleChange}
    //     color="secondary"
    //     name="volume"
    //     step={0.01}
    //     min={0}
    //     max={1}
    //   ></Slider>
    // </div>
  );
});
