import torch
import sys
from loguru import logger
from pathlib import Path
from tqdm import tqdm
import utils
import pandas as pd
import numpy as np
import librosa
import soundfile as sf
import uuid
import argparse
from models import crnn
import os

# edited by Maria Novakova for Bachelor Thesis

SAMPLE_RATE = 22050
EPS = np.spacing(1)
LMS_ARGS = {
    'n_fft': 2048,
    'n_mels': 64,
    'hop_length': int(SAMPLE_RATE * 0.02),
    'win_length': int(SAMPLE_RATE * 0.04)
}
DEVICE = 'cpu'
if torch.cuda.is_available():
    DEVICE = 'cuda'
DEVICE = torch.device(DEVICE)


def extract_feature(wavefilepath, **kwargs):
    _, file_extension = os.path.splitext(wavefilepath)
    if file_extension == '.wav':
        wav, sr = sf.read(wavefilepath, dtype='float32')
    if file_extension == '.mp3':
        wav, sr = librosa.load(wavefilepath)
    elif file_extension not in ['.mp3', '.wav']:
        raise NotImplementedError('Audio extension not supported... yet ;)')
    if wav.ndim > 1:
        wav = wav.mean(-1)
    wav = librosa.resample(wav, sr, target_sr=SAMPLE_RATE)
    return np.log( librosa.feature.melspectrogram(wav.astype(np.float32), SAMPLE_RATE, ** kwargs) + EPS).T


class OnlineLogMelDataset(torch.utils.data.Dataset):
    def __init__(self, data_list, **kwargs):
        super().__init__()
        self.dlist = data_list
        self.kwargs = kwargs

    def __getitem__(self, idx):
        return extract_feature(wavefilepath=self.dlist[idx],
                               **self.kwargs), self.dlist[idx]

    def __len__(self):
        return len(self.dlist)


MODELS = {
    'sre': {
        'model': crnn,
        'outputdim': 2,
        'encoder': 'labelencoders/students.pth',
        'pretrained': 'sre/model.pth',
        'resolution': 0.02
    },
    'mlt': {
        'model': crnn,
        'outputdim': 2,
        'encoder': 'labelencoders/students.pth',
        'pretrained': 'mlt/model.pth',
        'resolution': 0.02
    },
    'ptt': {
        'model': crnn,
        'outputdim': 2,
        'encoder': 'labelencoders/students.pth',
        'pretrained': 'ptt/model.pth',
        'resolution': 0.02
    },
    'new': {
        'model': crnn,
        'outputdim': 2,
        'encoder': 'labelencoders/students.pth',
        'pretrained': 'new/model.pth',
        'resolution': 0.02
    },
}


def main():
    parser = argparse.ArgumentParser()
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument(
        '-w',
        '--wav',
        help=
        'A single wave/mp3/flac or any other compatible audio file with soundfile.read'
    )
    group.add_argument(
        '-l',
        '--wavlist',
        help=
        'A list of wave or any other compatible audio files. E.g., output of find . -type f -name *.wav > wavlist.txt'
    )
    parser.add_argument('-model', choices=list(MODELS.keys()), default='mlt')
    parser.add_argument(
        '--pretrained_dir',
        default='pretrained_models',
        help=
        'Path to downloaded pretrained models directory, (default %(default)s)'
    )
    parser.add_argument('-o',
                        '--output_path',
                        default=None,
                        help='Output folder to save predictions if necessary')
    parser.add_argument('-soft',
                        default=False,
                        action='store_true',
                        help='Outputs soft probabilities.')
    parser.add_argument('-hard',
                        default=False,
                        action='store_true',
                        help='Outputs hard labels as zero-one array.')
    parser.add_argument('-th',
                        '--threshold',
                        default=(0.5, 0.1),
                        type=float,
                        nargs="+")
    args = parser.parse_args()
    pretrained_dir = Path(args.pretrained_dir)
    if not (pretrained_dir.exists() and pretrained_dir.is_dir()):
        logger.error(f"""Pretrained directory {args.pretrained_dir} not found.
Please download the pretrained models from and try again or set --pretrained_dir to your directory."""
                     )
        return
    logger.info("Passed args")
    for k, v in vars(args).items():
        logger.info(f"{k} : {str(v):<10}")
    if args.wavlist:
        wavlist = pd.read_csv(args.wavlist,
                              usecols=[0],
                              header=None,
                              names=['filename'])
        wavlist = wavlist['filename'].values.tolist()
    elif args.wav:
        wavlist = [args.wav]
    dset = OnlineLogMelDataset(wavlist, **LMS_ARGS)

    dloader = torch.utils.data.DataLoader(dset,
                                          batch_size=1,
                                          num_workers=3,
                                          shuffle=False)

    model_kwargs_pack = MODELS[args.model]
    model_resolution = model_kwargs_pack['resolution']
    # Load model from relative path

    model = model_kwargs_pack['model'](
        outputdim=model_kwargs_pack['outputdim'],
        pretrained_from=pretrained_dir /
        model_kwargs_pack['pretrained']).to(DEVICE).eval()
        

    encoder = torch.load(pretrained_dir / model_kwargs_pack['encoder'])
    print(model_kwargs_pack['encoder'])
    logger.trace(model)

    output_dfs = []
    frame_outputs = {}
    threshold = tuple(args.threshold)
#
    speech_label_idx = np.where('Speech' == encoder.classes_)[0].squeeze()

    # Using only binary thresholding without filter
    if len(threshold) == 1:
        postprocessing_method = utils.binarize
    else:
        postprocessing_method = utils.double_threshold

    ptt_predictions = []
#
    with torch.no_grad(), tqdm(total=len(dloader), leave=False, unit='clip') as pbar:
        for feature, filename in dloader:

            feature = torch.as_tensor(feature).to(DEVICE)



            prediction_tag, prediction_time, prediction_tag_ptt, prediction_time_ptt = model(feature)

            prediction_tag = prediction_tag.to('cpu')
            prediction_time = prediction_time.to('cpu')
            prediction_tag_ptt = prediction_tag_ptt.to('cpu')
            prediction_time_ptt = prediction_time_ptt.to('cpu')


            threshold_ptt_array = np.zeros(len(prediction_time_ptt[0]))

            start = []
            first = False
            end = []
            counter = 0
            
            for i in range(len(prediction_time_ptt[0])):
                
                # plot image in the bachelor thesis has for ptt set threshold 0.8
                # without threshold it is suitable when generating both models
                if prediction_time_ptt[0][i][0] < prediction_time_ptt[0][i][1]:

                    if first == False:
                        start.append(i)
                    first = True

                    threshold_ptt_array[i] = 1
                    counter += 1
                else:
                    if counter != 0:
                        end.append(i-1)
                    counter = 0
                    first = False


            ptt_pred_label_df = pd.DataFrame()
            
            ptt_pred_label_df['onset'] = (np.array(start)*441/SAMPLE_RATE).tolist()
            ptt_pred_label_df['offset'] = (np.array(end)*441/SAMPLE_RATE).tolist()
            ptt_pred_label_df['filename'] = filename[0]

            ptt_predictions.append(ptt_pred_label_df)

  
            if prediction_time is not None:  # Some models do not predict timestamps
                cur_filename = filename[0]  #Remove batchsize
                thresholded_prediction = postprocessing_method(prediction_time, *threshold)

                labelled_predictions = utils.decode_with_timestamps(encoder, thresholded_prediction)

                pred_label_df = pd.DataFrame(
                    labelled_predictions[0],
                    columns=['event_label', 'onset', 'offset'])
                
                if not pred_label_df.empty:
                    pred_label_df['filename'] = cur_filename
                    pred_label_df['onset'] *= model_resolution
                    pred_label_df['offset'] *= model_resolution
                    pbar.set_postfix(labels=','.join(np.unique(pred_label_df['event_label'].values)))
                    pbar.update()
                    output_dfs.append(pred_label_df)

    print("SPEECH PREDICTIONS:")

    full_prediction_df = pd.concat(output_dfs).sort_values(by='onset',ascending=True).reset_index()
    prediction_df = full_prediction_df[full_prediction_df['event_label'] == 'Speech']

    print(prediction_df.to_markdown(index=False))

    print("PUSH TO TALK PREDICTIONS:")

    full_prediction_df = pd.concat(ptt_predictions).sort_values(by='onset',ascending=True).reset_index()

    print(full_prediction_df.to_markdown(index=False))

if __name__ == "__main__":
    main()
