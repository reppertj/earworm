import React, { useState } from 'react';
import makeStyles from '@material-ui/core/styles/makeStyles';
import FormControl from '@material-ui/core/FormControl';
import FormLabel from '@material-ui/core/FormLabel';
import RadioGroup from '@material-ui/core/RadioGroup';
import Radio from '@material-ui/core/Radio';
import FormControlLabel from '@material-ui/core/FormControlLabel';

import { regionColorsMui } from './UploadSurfer';

const useStyles = makeStyles(theme => ({
  radio: { marginLeft: '0.2em', paddingTop: '0.05em', paddingBottom: '0.05em' },
  root: { paddingTop: '0.05em', paddingBottom: '0.05em' },
  formLabel: { paddingLeft: '0.2em', paddingTop: '0.2em' },
  label: { fontSize: 'small' },
}));

interface RegionPreferencesProps {
  regionID: number;
  regionPreference: number;
  updateRegionPreference: (regionID: any, newPreference: any) => void;
}

export default function RegionPreferences(props: RegionPreferencesProps) {
  const { regionID, regionPreference, updateRegionPreference } = props;
  const [preference, setPreference] = useState(regionPreference);

  const color = regionColorsMui[regionID - 1];

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
    const newPreference = preferencesMap[ev.target.value];
    updateRegionPreference(regionID, newPreference);
    setPreference(newPreference);
  };

  return (
    <>
      <FormControl component="fieldset">
        <FormLabel className={classes.formLabel} component="legend">
          Zone {regionID} Focus:
        </FormLabel>
        <RadioGroup value={value} onChange={handleChange}>
          {Object.keys(preferencesMap).map(desc => (
            <FormControlLabel
              classes={{ root: classes.root, label: classes.label }}
              key={desc}
              value={desc}
              control={
                <Radio
                  color={color}
                  classes={{ root: classes.radio }}
                  size="small"
                />
              }
              label={desc}
            />
          ))}
        </RadioGroup>
      </FormControl>
    </>
  );
}
