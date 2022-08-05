# Automated Speech Recognition (ASR)

[![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)](https://www.python.org/)
[![tf](https://img.shields.io/badge/Tensorflow-2.4+-a.svg)](https://www.tensorflow.org/)

This Chinese ASR project provides a framework to train and deploy a deep learning-based speech recognition model based on open source mandarin audio data. Two different modes is developed for users to choose:

**Mode A** models can recognize pinyins from audio and extra language models are needed for predicting characters from a sequence of pinyins. 

**Mode C** models is a end-to-end model which can predict Chinese characters from audio directly.

## Installation Guideline

This method is tested based on [Ubuntu 20.04](https://releases.ubuntu.com/20.04/). For Mac and Windows users, please find other methods to install the required dependencies. 

### Install FFmepg
```
sudo apt update

sudo apt install -y ffmpeg
```

### Create and activate virtual environment

Using an virtual environment is recommended for all individual Python projects. Since [Conda](https://docs.anaconda.com/anaconda/install/linux/) is used in developing this project, it would be used in demostration. Users can also use *virtualenv* function to create the virtual environment.

```
conda create -n venvasr python=3.8.10
conda activate venvasr
```

To see more about Conda commands: [conda cheatsheet](https://docs.conda.io/projects/conda/en/4.6.0/_downloads/52a95608c49671267e40c689e0bc00ca/conda-cheatsheet.pdf)

### Install python dependices

```
pip install -r requirements.txt
```

## Configuration

In [config.py](config.py), users can set the configuration of the ASR project, such as the path of the audio folder, the version of trained model, or the dataset selected, etc. Be careful that wrong configuration may lead to unexpected behaviour or raising error. 

## Dataset

Refer to the [audio folder](https://drive.google.com/file/d/1ATwJj9YkSof0kIOjHta4YCymlbywPPIS/view?usp=sharing) (Only included the data label, audio data need to be downloaded separately)

### Usage:

1. Unzip the downloaded file
2. Modify the data path in [config.py](config.py)
3. Follow the instruction in audio_folder/README.md

## Pre-trained model

A folder included the pre-trained model verR1.1c can be download from [here](https://drive.google.com/file/d/1-vhyLZMFA6welUwXFiLerDwA3f3THqm8/view?usp=sharing).

### Usage:

1. Unzip the downloaded file
2. Modify the speech_model_folder in [config.py](config.py)

## Run

> For all the files mentioned in this part, add option --help for getting the help messages

### Train the base model:

Run the [train_model.py](train_model.py) to start training a new model. The detail of training would be refered to [config.py](config.py). Training can be continued from saved weight or other models. If option --load_weight is not specified, users can control the program in the Command-Line Interface (CLI). 
```
python train_mode.py [option]

## Example:
python train_model.py --lr 0.001
```

### Train the language model:

For mode A model, an extra language model is necessary for translating the pinyin result to character. To train or test the language model, run the [train_language.py](train_model.py). Users can control the actions in the Command-Line Interface (CLI).
```
python train_language.py
```

### Usage example:

[example.py](example.py) is a use case of using the asr model and outputting the result on CLI. Make sure the configuration is the same as the training process of the model selected. 
```
python example.py [option]

## Example:
python example.py --path /some/path/to/audio.wav --version verR1.0c
```

### Performance:

[performance.py](performance.py) can be used to calculate the Word Error Rate (WER) of a single model or more than one models for comparsion.
```
python performance.py [option] [versions]

## Example:
python performance.py -mode dev -epoch 300 verR1.0c verR2.0c
```

## Tensorflow serving

The trained model can run in tensorflow serving container with simple steps:

**Pre-requisite: Docker is installed correctly**

1. Pull the docker image

    ```
    docker pull tensorflow/serving:latest
    ```

2. Change the current directory to your speech_model_folder, and run:

    ```
    mkdir ModeC
    cp -r verR1.1c/PINYIN_verR1.1c_base ModeC/1
    ```

3. Run the tensorflow serving with changing the path of speech_model_folder:

    ```
    docker run -t --rm -p 8501:8501 \
    -v "~/your_wd/model_PINYIN/ModeC:/models/asr" \
    -e MODEL_NAME=asr \
    tensorflow/serving
    ```

4. Set USE_TFX = True in [config.py](config.py)

## Credit

This project start from the referring to [nl8590687's ASRT project](https://github.com/nl8590687/ASRT_SpeechRecognition). Thank you so much for the learning material and concept provided. 