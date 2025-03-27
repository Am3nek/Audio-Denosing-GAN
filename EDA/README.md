## **Introduction** <br />
This is the Exploratory Data Analysis which comapres the basic features of the trainset and testset of both noisy and clean dataset. In this five audio files have been take in which the following have been performed -
1. **Feature Extraction :** The feature extraction in numerical values, representing the audio signals. Each number corresponds to a specific feature extracted from the audio signal.
2. **MFCC :** The MFCC capture the shape of the sound spectrum. It extract important frequency characteristics from an audio signal by mimicking how the human ear perceives sound and plots the graph.
3. **Mel Spectrogram :** It is a time-frequency representation of an audio signal, where the frequencies are converted to the Mel scale to better match human hearing perception.
4. **Spectral Bandwidth :** It measures the spread of frequencies in an audio signal. It helps determine how "wide" or "narrow" the frequency content is. A higher spectral bandwidth means the signal contains a broader range of frequencies, while a lower spectral bandwidth indicates that the signal is more concentrated in a smaller frequency range.
5. **Spectral Centroid :** It represents the "center of mass" of the spectrum, indicating where most of the frequency energy is concentrated in an audio signal.
6. **Spectral Contrast :** It measures the difference between the peaks (high energy) and valleys (low energy) in the frequency spectrum of an audio signal. It highlights the variations in energy across different frequency bands.
7. **Spectral Flatness :** It is a measure of how noise-like or tonal a sound is. It quantifies how evenly the energy is distributed across the frequency spectrum. High Spectral Flatness means the spectrum is flat, meaning the sound is noise-like. Low Spectral Flatness means the spectrum has distinct peaks, meaning the sound is tonal.
8. **Waveform :** The graphical representation shows how the amplitude (loudness) of the sound varies as a function of time.
9. **Zero Crossing Rate :** It measures how many times the audio waveform crosses the zero-amplitude line per unit time. It is a key feature for distinguishing between different types of sounds. High ZCR means more zero crossings and low ZCR means fewer zero crossings.


We perform these analysis to understand the difference between the noisy and clean audios.
