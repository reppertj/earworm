import {
  createSlice as createSliceOriginal,
  SliceCaseReducers,
  CreateSliceOptions,
} from '@reduxjs/toolkit';
import { RootState } from '../../types/RootState';

/* Wrap createSlice with stricter Name options */

type RootStateKeyType = keyof RootState;

/* istanbul ignore next */
export const createSlice = <
  State,
  CaseReducers extends SliceCaseReducers<State>,
  Name extends RootStateKeyType
>(
  options: CreateSliceOptions<State, CaseReducers, Name>,
) => {
  return createSliceOriginal(options);
};
