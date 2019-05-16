# 'Recept' - Wake word Detection

We are going to construct a audio speech dataset and implement an algorithm for Wake word detection which also called as keyword detection, or wakeword detection. Trigger word detection is the technology that allows devices like Amazon Alexa, Google Home, Apple Siri, etc to wake up, turn on upon hearing a certain word.

Our trigger word will be "Activate." so every time you say "activate," the device is going to make a "chiming" sound. We will also be able to record a clip of yourself talking in the audio microphone, and have the algorithm trigger a chiming sound when it detects saying "activate."

In this project we are going to acheive 
* Structure a speech recognition project
* Synthesize and process audio recordings to create train/dev datasets
* Train a trigger word detection model and make predictions

## Motivation
![](https://github.com/Ruthvicp/CS5590_PyDL/blob/master/Module1/Lab_Assignment/Lab2/documentation/motivation.png)

Amazon Alexa, Google Voice Assistant, Apple Siri are biult on high level GPU's and trained on over 250 million hours of audio recordings, which is not open source and non Customisable

We have biult the trigger word dettection on a light weight keras based model which is open source and can be easily deployed on a embedded device.

![](https://github.com/Ruthvicp/CS5590_PyDL/blob/master/Module1/Lab_Assignment/Lab2/documentation/motivation2.png)

![](https://github.com/Ruthvicp/CS5590_PyDL/blob/master/Module1/Lab_Assignment/Lab2/documentation/motivation3.png)

## Project Idea

### Data Preprocessing : Creating a speech Dataset
Start by building a dataset for your trigger word detection algorithm, We need to create recordings with a combination of positive words (in our case "activate") and negative words (random words other than "activate") on different background sounds.

We started by recording 10 second audio clips with varying accents containing the trigger word of about 4000 samples.

### Audio Recordings to Spectograms
In order to help our sequence model more easily learn to detect triggerwords, we will need to compute a spectrogram of the audio. The spectrogram tells us how much different frequencies are present in an audio clip at a moment in time.

A spectrogram is computed by sliding a window over the raw audio signal, and calculates the most active frequencies in each window using a Fourier transform.

![](https://github.com/Ruthvicp/CS5590_PyDL/blob/master/Module1/Lab_Assignment/Lab2/documentation/spectogram.png)
The graph above represents how active each frequency is (y axis) over a number of time-steps (x axis).
### Generating a training Sample

With 10 seconds being our default training example length, 10 seconds of time can be discretized to different numbers of value. We have observed 441000 (raw audio) and 5511 (spectrogram). In the former case, each step represents 10/441000 approx 0.000023 seconds. In the second case, each step represents 10/5511 approx 0.0018 seconds.

For the 10sec of audio, the key values you will see in this assignment are:

$441000 raw audio
$5511 = T_x which is the spectrogram output, and dimension of input to the neural network
10000 used by the pydub module to synthesize audio
$1375 = T_y which is the number of steps in the output of the GRU you'll build

To obtain a single training example we need:

* Pick a random 10 second background audio clip
* Randomly insert 0-4 audio clips of "activate" into this 10sec clip
* Randomly insert 0-2 audio clips of negative words into this 10sec clip

### Full training set
After we implement the code needed to generate a single training example. We used this process to generate a large training set. To save time, we've already generated a set of training examples.

Our dataset now consists of 4000 audio samples with 2 positives and 1 negative word for each audio clip.
We one hot encode the positive values with 1's and negative words and background noise with 0's

## System Architecture
![](https://github.com/Ruthvicp/CS5590_PyDL/blob/master/Module1/Lab_Assignment/Lab2/documentation/architecture.svg)

The 1D convolutional step inputs 5511 timesteps of the spectrogram 10 seconds audio file in our case, outputs a 1375 step output. It extracts low-level audio features similar to how 2D convolutions extract image features. Also helps speed up the model by reducing the number of timesteps.

The two GRU layers read the sequence of inputs from left to right, then ultimately uses a dense+sigmoid layer to make a prediction. Sigmoid make the range of each label between 0~1. Being 1, corresponding to the user having just said "activate".

## Technical Stack
We have used the following technical specs for this project

* Python 3.4 and above
* Pydub package for reading audio files 
* Jupyter notebook
To implement Metrics and plot graphs
* numpy
* keras
* h5py
* pydub
* scipy
* matplotlib

## Implementation

## Contribution

## References

