import tensorflow.keras as keras
import numpy as np
import librosa
import os

MODEL_PATH = "model.h5"
NUM_SAMPLE_TO_CONSIDER = 22050

class _Keyword_Spotting_Service:

    model = None
    _mapping = [
        "down",
        "go",
        "happy",
        "left",
        "no",
        "off",
        "on",
        "right",
        "stop",
        "up",
        "yes"
    ]
    _instance = None


    def predict(self, file_path):
        # extract MFCCs
        MFCCs = self.preprocess(file_path)  # (# segments,# coefficients)

        if isinstance(MFCCs, np.ndarray):  # checks if data being sent is good

            # convert 2d MFCCs array into 4d Array --> (#samples, #segments, # coefficients, # channels = 1)
            MFCCs = MFCCs[np.newaxis, ..., np.newaxis]

            # make prediction
            predictions = self.model.predict(MFCCs)
            predicted_index = np.argmax(predictions)
            predicted_keyword = self._mapping[predicted_index]

            return predicted_keyword
        else:
            return "file too small"

    def preprocess(self, file_path, n_mfcc=13, n_fft=2048, hop_length=512):

        # load audio file
        signal, sr = librosa.load(file_path)

        # ensure consistency in audio file length
        if len(signal) >= NUM_SAMPLE_TO_CONSIDER:
            signal = signal[:NUM_SAMPLE_TO_CONSIDER]
            MFCCs = librosa.feature.mfcc(signal, n_mfcc=n_mfcc, n_fft=n_fft, hop_length=hop_length)
            return MFCCs.T
        # extract MFCCs

        return "bad"  # in case the audio file is too small


def Keyword_Spotting_Service():  # this is function is used to implement a singleton class
    # ensure that we only have 1 instance
    if _Keyword_Spotting_Service._instance is None:
        _Keyword_Spotting_Service._instance = _Keyword_Spotting_Service()
        _Keyword_Spotting_Service.model = keras.models.load_model(MODEL_PATH)
    return _Keyword_Spotting_Service._instance

if __name__ == "__main__":
    kss = Keyword_Spotting_Service()

    #print(kss.predict("yes.wav"))
    for subdir, dirs, files in os.walk("Predict"):

        # iterating through files
        for file in files:
            file_path = subdir + os.sep + file
            print(f" {file}: ", kss.predict(file_path))


