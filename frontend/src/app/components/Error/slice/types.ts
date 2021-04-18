import { EntityState } from '@reduxjs/toolkit';

/* --- STATE --- */
export interface Error {
  id: string;
  error: boolean;
  message: string;
}

export type ErrorState = EntityState<Error>;
