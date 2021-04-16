/* --- STATE --- */
export interface SpotifyAuthState {
  loggedIn: boolean;
  accessToken: string | undefined;
  tokenExpiryDate: number | undefined;
}
