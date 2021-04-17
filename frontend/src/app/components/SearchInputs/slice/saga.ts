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
import getChannelDataAndSampleRate from 'app/utils/audio';

function* revokeObjectUrl({ payload }: { payload: Source[] }) {
  for (const source of payload) {
    const assigned = yield select(selectSourceByID, source.id);
    if (assigned) {
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
    }
  }
}

export function* audioSourcesSaga() {
  yield takeEvery(actions.sourcesAddedIfRoom, revokeObjectUrl);
}
