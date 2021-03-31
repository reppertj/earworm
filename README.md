# Earworm
> Search for royalty-free music by sonic similarity

Earworm is a search engine for music licensed for royalty-free commercial use, where you search by sonic similarity to something you already know. Either upload some mp3s or connect your Spotify account, then use clips from songs you like to search for similar songs that are licensed for you to use in your next great video game or YouTube video (always verify the license terms first; many licenses still require attribution!). You can search for the overall closest fit, or choose to focus matching genre, mood, or instrumentation. All the audio processing is done in your browser; your music never leaves your device.

![A 2D (UMAP) projection of some music embeddings, colored by mood](embeddings_by_mood.png)
> A 2d projection of some embeddings, in this case colored by mood

## Status

This is a *Work in Progress*! While the modeling is complete, the webapp is actively under development and still has many incomplete features.

## Installation

OS X & Linux:

All models can easily fit on a 16GB GPU; the final model is based on a MobileNetV2 architecture, though the training loop is somewhat involved, using a MoCo-style queue, a combination of fully supervised and self-supervised objectives, and class-aware sampling to increase the number of informative comparisons per batch. The main training loop definition is in [modeling/music_metric_learning/models/train_music_learning.py](modeling/music_metric_learning/models/train_music_learning.py). A Jupyter Notebook illustrating some of these training dynamics is in progress.

To setup the modeling environment using Conda:

```sh
cd modeling
conda env create -f environment.tml
```

## Meta

Distributed under the MIT license. See ``LICENSE`` for more information.
