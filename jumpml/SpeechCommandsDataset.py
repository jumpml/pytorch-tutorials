import torch
import torchaudio
from collections import defaultdict
import os
import shutil
import glob

KNOWN_COMMANDS = ["yes",
                  "no",
                  "up",
                  "down",
                  "left",
                  "right",
                  "on",
                  "off",
                  "stop",
                  "go",
                  "background"]


def get_fileList(dataset_path, filename):
    with open(os.path.join(dataset_path, filename)) as f:
        fileList = f.read().split("\n")

    fileList = [os.path.join(dataset_path, fname) for fname in fileList]
    return(fileList)


def generate_background_files(dataset_path, num_samples=16000):

    background_source_files = glob.glob(os.path.join(
        dataset_path, "_background_noise_", "*.wav"))

    targetDir = os.path.join(dataset_path, "background")
    # Generate Background Files
    print('Generate 1s background files:\n')
    os.makedirs(targetDir, exist_ok=True)
    for f in background_source_files:
        waveform, sr = torchaudio.load(f)
        split_waveforms = torch.split(waveform, num_samples, dim=1)
        for idx, split_waveform in enumerate(split_waveforms):
            torchaudio.save(os.path.join(
                targetDir, f'{hash(waveform)}_nohash_{idx}.wav'), split_waveform, sample_rate=sr)

    background_target_files = glob.glob(
        os.path.join(targetDir, "*.wav"))
    return(background_target_files)


class SpeechCommandsData:
    def __init__(self, path='.', train_bs=64, test_bs=256, val_bs=64, n_mels=40):

        # Setup transforms (separate for train, test and val if necessary)
        self.transform = torch.nn.Sequential(
            torchaudio.transforms.MelSpectrogram(sample_rate=16000, n_fft=256, hop_length=128, n_mels=n_mels),
            torchaudio.transforms.AmplitudeToDB(stype='power', top_db=80))

        # Cleanup background files if needed
        backgroundDir = os.path.join(
            path, 'SpeechCommands', 'speech_commands_v0.02', 'background')
        if (os.path.isdir(backgroundDir)):
            print(
                'Found existing background directory. Removing files in that directory.')
            shutil.rmtree(backgroundDir)

        # Create separate datasets (or filelist iterators) for train, test and val
        print('Initialize/download SpeechCommandsDataset....\n')
        self.train_dataset = SpeechCommandsDataset(
            path=path, transform=self.transform)
        self.val_dataset = SpeechCommandsDataset(
            path=path, transform=self.transform)
        self.test_dataset = SpeechCommandsDataset(
            path=path, transform=self.transform)

        self.dataset_len = len(self.train_dataset)
        print(f'SpeechCommands Dataset Size: {self.dataset_len}\n')

        # Generate background files: creates files with 1 sec duration
        self.background_fileList = generate_background_files(
            self.train_dataset.dataset._path)
        print(f'Background files generated: {len(self.background_fileList)}\n')

        # Get validation and test file list from file
        self.val_fileList = get_fileList(
            self.val_dataset.dataset._path, "validation_list.txt")
        self.test_fileList = get_fileList(
            self.test_dataset.dataset._path, "testing_list.txt")

        self.val_ratio = len(self.val_fileList) / self.dataset_len
        self.test_ratio = len(self.test_fileList) / self.dataset_len
        self.train_ratio = 1.0 - self.val_ratio - self.test_ratio

        # Filter out files: modify _walker
        print(f'Extracting training dataset files...')
        self.train_dataset.dataset._walker = list(
            filter(lambda x: x not in self.val_fileList
                   and x not in self.test_fileList,
                   self.train_dataset.dataset._walker)
        )
        print(f'Train dataset extracted: {len(self.train_dataset)} files \n')
        print(f'Extracting test and val dataset files...')
        self.val_dataset.dataset._walker = list(
            filter(lambda x: x in self.val_fileList,
                   self.val_dataset.dataset._walker))
        print(f'Validation dataset extracted: {len(self.val_dataset)} files')
        self.test_dataset.dataset._walker = list(
            filter(lambda x: x in self.test_fileList,
                   self.test_dataset.dataset._walker))
        print(f'Test dataset extracted: {len(self.test_dataset)} files')

        # Add background files to walker
        idx_train = int(self.train_ratio * len(self.background_fileList))
        self.train_dataset.dataset._walker += self.background_fileList[:idx_train]
        idx_val = idx_train + \
            int(self.val_ratio * len(self.background_fileList))
        self.val_dataset.dataset._walker += self.background_fileList[idx_train:idx_val]
        idx_test = idx_val + int(self.test_ratio *
                                 len(self.background_fileList))
        self.test_dataset.dataset._walker += self.background_fileList[idx_val:idx_test]

        # Create Dataloaders
        self.train_loader = torch.utils.data.DataLoader(
            self.train_dataset, batch_size=train_bs, shuffle=True)
        self.test_loader = torch.utils.data.DataLoader(
            self.test_dataset, batch_size=test_bs, shuffle=True)
        self.val_loader = torch.utils.data.DataLoader(
            self.val_dataset, batch_size=val_bs, shuffle=True)


class SpeechCommandsDataset(torch.utils.data.Dataset):
    def __init__(self, path='.', transform=None):
        self.dataset = torchaudio.datasets.SPEECHCOMMANDS(path,
                                                          url='speech_commands_v0.02',
                                                          folder_in_archive='SpeechCommands',
                                                          download=True)
        self.transform = transform

        # unknown word results in a default value of len(KNOWN_COMMANDS)
        self.word2num = defaultdict(lambda: len(KNOWN_COMMANDS)-1)
        for num, command in enumerate(KNOWN_COMMANDS):
            self.word2num[command] = num

    def __getitem__(self, index):
        (waveform, sample_rate, label, _, _) = self.dataset[index]
        # pad every waveform to 1 sec in samples
        padding = int((sample_rate - waveform.shape[1]))
        waveform = torch.nn.functional.pad(waveform, (0, padding))
        features = self.transform(waveform)
        label = self.word2num[label]

        return features, label

    def __len__(self):
        return len(self.dataset)
