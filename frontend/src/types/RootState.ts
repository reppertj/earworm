import { VolumeState } from 'app/components/Playback/volumeSlice/types';
import { SpotifyAuthState } from 'app/components/Spotify/slice/types';
// [IMPORT NEW CONTAINERSTATE ABOVE] < Needed for generating containers seamlessly

/* 
  Because the redux-injectors injects your reducers asynchronously somewhere in your code
  You have to declare them here manually
*/
export interface RootState {
  volume?: VolumeState;
  spotifyAuth?: SpotifyAuthState;
  // [INSERT NEW REDUCER KEY ABOVE] < Needed for generating containers seamlessly
}
