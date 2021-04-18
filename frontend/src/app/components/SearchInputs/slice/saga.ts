import { nanoid } from '@reduxjs/toolkit';
import {
  take,
  call,
  put,
  select,
  takeLatest,
  takeEvery,
} from 'redux-saga/effects';
import { selectSourceByID } from './selectors';
import { Source } from './types';
import { audioSourcesActions as actions } from '.';
import { errorActions } from './../../Error/slice';
import getChannelDataAndSampleRate from 'app/utils/audio';

function* preprocessAudioData({ payload }: { payload: Source[] }) {
  for (const source of payload) {
    const assigned = yield select(selectSourceByID, source.id);
    if (assigned) {
      try {
        const { channelData, sampleRate } = yield call(
          getChannelDataAndSampleRate,
          source.originalData,
        );
        console.log(actions.sourceUpdated.type);
        yield put(
          actions.sourceUpdated({
            id: source.id,
            changes: { ready: true, channelData, sampleRate },
          }),
        );
      } catch (e) {
        console.log(e);
        yield put(
          errorActions.errorAdded({
            id: nanoid(),
            error: true,
            message:
              'Error processing audio file. Check the file format, and make sure the clip is at least 5 seconds long.',
          }),
        );
        yield put(actions.sourceRemoved(source.id));
        yield call(URL.revokeObjectURL, source.originalData);
      }
    }
  }
}

export function* audioSourcesSaga() {
  yield takeEvery(actions.sourcesAddedIfRoom, preprocessAudioData);
}
