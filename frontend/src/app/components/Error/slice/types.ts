import { EntityState } from '@reduxjs/toolkit';

/* --- STATE --- */
export interface Error {
  error: boolean;
  message: string;
}

export type ErrorState = EntityState<Error>;
