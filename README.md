This project attempts to replicate Spotify's (and other music services') genre classifier and is modeled after Hareesh Bahuleyan's "Music Genre Classification Using Machine Learning Techniques." We obtained our data from the Free Music Archive (FMA), which was comprised of 8,000 audio files (1,000 each for eight genres). The FMA also provided feature data describing qualities of the songs (for example, acousticness, danceability, tempo).

We built two architectures to analyze the data. The first was a simple 2D CNN with Batch Normalization, Squeeze-Excitation, and Max Pooling. The second was the 2D CNN that ran in conjunction with dense layers (with Dropout) processing the tabular data. The data was concatenated after initial processing and then fed through more dense layers. 

In terms of preprocessing, the audio files (.mp3s) were used to generate spectrograms of the songs. We performed data augmentation by expanding on the high and low frequencies, so their differences were more discernable. These spectrograms were them converted into numpy arrays to be pushed through the 2D CNN.

With only the 2D CNN, we obtained an accuracy of 0.49. With the 2D CNN and tabular data, we obtained an accuracy of 0.51.
