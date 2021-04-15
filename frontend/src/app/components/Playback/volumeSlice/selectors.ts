import { RootState } from 'types';
import { initialState } from '.';

export const selectVolume = (state: RootState) => state?.volume || initialState;
