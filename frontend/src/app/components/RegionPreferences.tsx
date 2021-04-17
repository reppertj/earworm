import React, { useState } from 'react';
import makeStyles from '@material-ui/core/styles/makeStyles';
import FormControl from '@material-ui/core/FormControl';
import FormLabel from '@material-ui/core/FormLabel';
import RadioGroup from '@material-ui/core/RadioGroup';
import Radio from '@material-ui/core/Radio';
import FormControlLabel from '@material-ui/core/FormControlLabel';

import { useDispatch, useSelector } from 'react-redux';
import {
  selectRegionByID,
  selectRegionIdToSolidColorMap,
  selectRegionIdToShowIdMap,
} from './SearchInputs/Regions/slice/selectors';
import { useRegionsSlice } from './SearchInputs/Regions/slice';

interface RegionPreferencesProps {
  regionId: string;
}

export default function RegionPreferences(props: RegionPreferencesProps) {
  const dispatch = useDispatch();
  const { actions } = useRegionsSlice();
  const { regionId } = props;
  const region = useSelector(state => selectRegionByID(state, regionId));
  const [preference, setPreference] = useState(region?.preference);

  const colorMap = useSelector(selectRegionIdToSolidColorMap);
  const showIdMap = useSelector(selectRegionIdToShowIdMap);

  const showId = showIdMap[regionId];
  const color = colorMap[regionId];

  const useStyles = makeStyles(theme => ({
    radio: {
      marginLeft: '0.2em',
      paddingTop: '0.05em',
      paddingBottom: '0.05em',
      '&$checked': {
        color: color,
      },
    },
    checked: {
      color: color,
    },
    root: { paddingTop: '0.05em', paddingBottom: '0.05em' },
    formLabel: { paddingLeft: '0.2em', paddingTop: '0.2em' },
    label: { fontSize: 'small' },
  }));

  const classes = useStyles();

  const preferencesMap = {
    'Overall Sound': 4,
    Genre: 0,
    Instrumentation: 1,
    Mood: 2,
  };

  const value = Object.keys(preferencesMap).find(
    key => preferencesMap[key] === preference,
  );

  const handleChange = ev => {
    const newPreference: number = preferencesMap[ev.target.value];
    dispatch(
      actions.regionUpdated({
        id: regionId,
        changes: { preference: newPreference },
      }),
    );
    setPreference(newPreference);
  };

  return (
    <>
      <FormControl component="fieldset">
        <FormLabel className={classes.formLabel} component="legend">
          Zone {showId} Focus:
        </FormLabel>
        {value && (
          <RadioGroup value={value} onChange={handleChange}>
            {Object.keys(preferencesMap).map(desc => (
              <FormControlLabel
                classes={{ root: classes.root, label: classes.label }}
                key={desc}
                value={desc}
                control={
                  <Radio
                    color="primary"
                    classes={{ root: classes.radio, checked: classes.checked }}
                    size="small"
                  />
                }
                label={desc}
              />
            ))}
          </RadioGroup>
        )}
      </FormControl>
    </>
  );
}
