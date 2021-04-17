import { VolumeState } from 'app/components/Playback/volumeSlice/types';
import { SpotifyAuthState } from 'app/components/Spotify/slice/types';
import { AudioSourcesState } from 'app/components/SearchInputs/slice/types';
import { RegionsState } from 'app/components/SearchInputs/Regions/slice/types';
import { ErrorState } from 'app/components/Error/slice/types';
// [IMPORT NEW CONTAINERSTATE ABOVE] < Needed for generating containers seamlessly

/* 
  Because the redux-injectors injects your reducers asynchronously somewhere in your code
  You have to declare them here manually
*/
export interface RootState {
  volume?: VolumeState;
  spotifyAuth?: SpotifyAuthState;
  audioSources?: AudioSourcesState;
  regions?: RegionsState;
  error?: ErrorState;
  // [INSERT NEW REDUCER KEY ABOVE] < Needed for generating containers seamlessly
}
