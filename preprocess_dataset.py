import librosa
import os
import json


DATASET_PATH = "dataset"
JSON_PATH = "data.json"
SAMPLES_TO_CONSIDER = 22050
# 1 sec worth of sound. librosa's defarult sample rate is 22050
# since we know each clip is 1 sec, we care sonsidering 22050 samples.

# now we are preparing the mfccs and putiing them in a json file. Doing it in real time would be costly.
# Thus creating the json file.


def prepare_dataset(dataset_path, json_path, n_mfcc=13, hop_length= 512, n_fft= 2048):

    # data dictionary
    data = {
        "mapping": [],
        "labels": [],
        "MFCCs": [],
        "files": []
    }

    # loop through all the sub-directories
    for i, (dirpath, dirnames, filenames) in enumerate(os.walk(dataset_path)):

        # we need to ensure we are not at root level (ie, not looking at dataset_path)
        if dirpath is not dataset_path:

            # update mappings
            category = dirpath.split("/")[-1]  # split does : dataset/down --> [dataset, down]
            # data["mappings"].append(category)
            print(f"Processing {category}")

            # loop through all the filenames and extract MFCCs
            for f in filenames:

                # get file path
                file_path = os.path.join(dirpath, f)

                # load audio path
                signal, sr = librosa.load(file_path)

                # ensure the audio file is at least 1 sec
                if len(signal) >= SAMPLES_TO_CONSIDER:

                    # enforce 1 sec sound
                    # for cnn input should be of same length
                    # hence we only want a sound with 22050 samples
                    signal = signal[:SAMPLES_TO_CONSIDER]

                    # extract MFCCs
                    MFCCs = librosa.feature.mfcc(signal, n_mfcc=n_mfcc, hop_length=hop_length,
                                                 n_fft=n_fft)

                    # store data
                    data["labels"].append(i-1)
                    data["MFCCs"].append(MFCCs.T.tolist())
                    # librosa returns a ndarray but we are storing in a json
                    # thus transposing to a list
                    data["files"].append(file_path)
                    print(f"{file_path}: {i-1}")

    # store in json
    with open(json_path, "w") as fp:
        json.dump(data, fp, indent=4)


if __name__ == "__main__":
    prepare_dataset(DATASET_PATH, JSON_PATH)
