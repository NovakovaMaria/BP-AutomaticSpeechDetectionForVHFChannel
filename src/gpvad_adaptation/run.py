#!/usr/bin/env python
# -*- coding: utf-8 -*-

# edited by Maria Novakova for Bachelor Thesis

import os
import datetime

import uuid
import fire
from pathlib import Path

import pandas as pd
import torch
import numpy as np
from tqdm import tqdm
from ignite.contrib.handlers import ProgressBar, param_scheduler
from ignite.engine import (Engine, Events)
from ignite.handlers import EarlyStopping, ModelCheckpoint
from ignite.metrics import Accuracy, RunningAverage, Precision, Recall, ConfusionMatrix
from ignite.utils import convert_tensor
from tabulate import tabulate
from h5py import File

from sklearn.metrics import precision_recall_curve,average_precision_score, roc_curve, auc
import dataset
import models
import utils
import metrics
import losses
import torch.nn as nn

DEVICE = 'cpu'
if torch.cuda.is_available(
) and 'SLURM_JOB_PARTITION' in os.environ and 'gpu' in os.environ[
        'SLURM_JOB_PARTITION']:
    DEVICE = 'cuda'
    # Without results are slightly inconsistent
    torch.backends.cudnn.deterministic = True
DEVICE = torch.device(DEVICE)

acc = 0.0
def accfunc(a):
    global acc
    if a > acc:
        return True
    return False

class Runner(object):
    """Main class to run experiments with e.g., train and evaluate"""
    def __init__(self, seed=42):
        """__init__

        :param config: YAML config file
        :param **kwargs: Overwrite of yaml config
        """
        super().__init__()
        torch.manual_seed(seed)
        np.random.seed(seed)

    @staticmethod
    def _forward(model, batch):
        
        inputs, targets_time, targets_clip, ptt_targets_time, ptt_targets_clip, filenames, lengths = batch

        inputs = convert_tensor(inputs, device=DEVICE, non_blocking=True)
#
        targets_time = convert_tensor(targets_time,
                                      device=DEVICE,
                                      non_blocking=True)
        targets_clip = convert_tensor(targets_clip,
                                      device=DEVICE,
                                      non_blocking=True)
        
        ptt_targets_time = convert_tensor(ptt_targets_time,
                                            device=DEVICE,
                                            non_blocking=True)
        ptt_targets_clip = convert_tensor(ptt_targets_clip,
                                            device=DEVICE,
                                            non_blocking=True)
        
        clip_level_output, frame_level_output, ptt_clip_level_output, ptt_level_output = model(inputs)


        return clip_level_output, frame_level_output, targets_time, targets_clip, lengths,\
                ptt_clip_level_output, ptt_level_output, ptt_targets_time, ptt_targets_clip, lengths

    @staticmethod
    def _negative_loss(engine):
        return -engine.state.metrics['Loss']

    def train(self, config, **kwargs):
        """Trains a given model specified in the config file or passed as the --model parameter.
        All options in the config file can be overwritten as needed by passing --PARAM
        Options with variable lengths ( e.g., kwargs can be passed by --PARAM '{"PARAM1":VAR1, "PARAM2":VAR2}'

        :param config: yaml config file
        :param **kwargs: parameters to overwrite yaml config
        """

        # START OF CONFIGURATION AND PREPARATION OF DATASET
        config_parameters = utils.parse_config_or_kwargs(config, **kwargs)
        outputdir = os.path.join(
            config_parameters['outputpath'], config_parameters['model'],
            "{}_{}".format(
                datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%m'),
                uuid.uuid1().hex))
        
        # Early init because of creating dir
        checkpoint_handler = ModelCheckpoint(
            outputdir,
            'run',
            n_saved=3,
            require_empty=False,
            create_dir=True,
            score_function=self._negative_loss,
            score_name='loss')
        
        logger = utils.getfile_outlogger(os.path.join(outputdir, 'train.log'))
        logger.info("Storing files in {}".format(outputdir))

        # utils.pprint_dict
        utils.pprint_dict(config_parameters, logger.info)
        logger.info("Running on device {}".format(DEVICE))
        print(config_parameters['label'])
        label_df = pd.read_csv(config_parameters['label'], sep='\s+')
        data_df = pd.read_csv(config_parameters['data'], sep='\s+')

        # In case that both are not matching
        merged = data_df.merge(label_df, on='filename')
        common_idxs = merged['filename']
        data_df = data_df[data_df['filename'].isin(common_idxs)]
        label_df = label_df[label_df['filename'].isin(common_idxs)]

        train_df, cv_df = utils.split_train_cv(label_df, **config_parameters['data_args'])
        train_label = utils.df_to_dict(train_df)
        cv_label = utils.df_to_dict(cv_df)
        data = utils.df_to_dict(data_df)

        transform = utils.parse_transforms(config_parameters['transforms'])


        logger.info("Transforms:")
        utils.pprint_dict(transform, logger.info, formatter='pretty')
        assert len(cv_df) > 0, "Fraction a bit too large?"
            
        trainloader = dataset.gettraindataloader(
            h5files=data,
            h5labels=train_label,
            transform=transform,
            label_type=config_parameters['label_type'],
            batch_size=config_parameters['batch_size'],
            num_workers=config_parameters['num_workers'],
            shuffle=True,
        )

        cvdataloader = dataset.gettraindataloader(
            h5files=data,
            h5labels=cv_label,
            label_type=config_parameters['label_type'],
            transform=None,
            shuffle=False,
            batch_size=config_parameters['batch_size'],
            num_workers=config_parameters['num_workers'],
        )

        # END OF CONFIGURATION AND PREPARATION OF DATASET

        # MODEL LOADING
        model = getattr(models, config_parameters['model'],
                        'CRNN')(inputdim=trainloader.dataset.datadim,
                                outputdim=2,
                                **config_parameters['model_args'])
        
        if 'pretrained' in config_parameters and config_parameters['pretrained'] is not None:
            model_dump = torch.load(config_parameters['pretrained'],map_location='cpu')
            model_state = model.state_dict()
            pretrained_state = {
                k: v
                for k, v in model_dump.items()
                if k in model_state and v.size() == model_state[k].size()
            }
            model_state.update(pretrained_state)
            model.load_state_dict(model_state)
            logger.info("Loading pretrained model {}".format(config_parameters['pretrained']))


        model = model.to(DEVICE)

        for name, param in model.named_parameters():
            if 'outputlayer1' not in name:
                print("Not trainable", name)
                param.requires_grad = False
            else:
                print("Trainable", name)
                param.requires_grad = True

        optimizer = getattr(torch.optim, config_parameters['optimizer'],)(model.parameters(), **config_parameters['optimizer_args'])


        utils.pprint_dict(optimizer, logger.info, formatter='pretty')
        utils.pprint_dict(model, logger.info, formatter='pretty')

        if DEVICE.type != 'cpu' and torch.cuda.device_count() > 1:
            logger.info("Using {} GPUs!".format(torch.cuda.device_count()))
            model = torch.nn.DataParallel(model)

        criterion = getattr(losses, config_parameters['loss'])().to(DEVICE)

        def _train_batch(_, batch):
            model.eval()

            with torch.enable_grad():
                optimizer.zero_grad()

                output = self._forward(model, batch)  # output is tuple (clip, frame, target)

                lossPTT = output[5:]   
                loss = criterion(*lossPTT) 

                loss.backward()
                #Single loss
                optimizer.step()

            return loss.item()

        def _inference(_, batch):
            model.eval()
            with torch.no_grad():
                return self._forward(model, batch)
        
        
        def thresholded_output_transform(output):
            # Output is (clip, frame, target, lengths)
            _, y_pred, y, y_clip,length, _, ptt_y_pred, ptt, ptt_y_clip, length = output
            
            batchsize, timesteps, ndim = y.shape

            idxs = torch.arange(timesteps, device='cpu').repeat(batchsize).view(batchsize, timesteps)

            mask = (idxs < length.view(-1, 1)).to(y.device)

            y = y * mask.unsqueeze(-1)

            y_pred = torch.round(y_pred)

            y = torch.round(y)

            return y_pred, y
        #        
        def thresholded_output_transformPTT(output):
            # Output is (clip, frame, target, lengths)
            _, y_pred, y, y_clip,length, clip, ptt_y_pred, ptt, ptt_y_clip, length = output
            
            batchsize, timesteps, ndim = ptt.shape

            idxs = torch.arange(timesteps, device='cpu').repeat(batchsize).view(batchsize, timesteps)

            mask = (idxs < length.view(-1, 1)).to(ptt.device)

            ptt = ptt * mask.unsqueeze(-1)

            ptt_y_pred = torch.round(ptt_y_pred)

            return ptt_y_pred, ptt
        
        metrics = {
            'Loss': losses.Loss(criterion),  #reimplementation of Loss, supports 3 way loss 
            'Precision': Precision(thresholded_output_transform),
            'Recall': Recall(thresholded_output_transform),
            'Accuracy': Accuracy(thresholded_output_transform),
            'PrecisionPTT': Precision(thresholded_output_transformPTT),
            'RecallPTT': Recall(thresholded_output_transformPTT),
            'AccuracyPTT': Accuracy(thresholded_output_transformPTT),
        }

        train_engine = Engine(_train_batch)
        inference_engine = Engine(_inference)

        for name, metric in metrics.items():
            metric.attach(inference_engine, name)

        def compute_metrics(engine):

            inference_engine.run(cvdataloader)

            results = inference_engine.state.metrics

            output_str_list = ["Validation Results - Epoch : {:<5}".format(engine.state.epoch)]
            
            update = [False, False, False]

            accuracy = 0
            recall = 0
            precision = 0 

            totalacc = 0

            for metric in metrics:
                output_str_list.append("{} {:<5.2f}".format(metric, results[metric]))

                if metric == "AccuracyPTT" and accuracy <= results[metric]:
                    update[0] = True
                    accuracy = results[metric]
                    totalacc += accuracy
                elif metric == "RecallPTT" and recall <= results[metric]:
                    update[1] = True
                    recall = results[metric]
                elif metric == "PrecisionPTT" and precision <= results[metric]:
                    update[2] = True
                    precision = results[metric]

                if  metric == "Accuracy":
                    totalacc += results[metric]

            
            print("finished, accuracy", totalacc/2)

            if update[0] and update[1] and update[2]:
                print("saving the model..")
                torch.save(model.state_dict(), os.path.join(outputdir,'model.pth'))
                update = [False, False, False]

            logger.info(" ".join(output_str_list))

            pbar.n = pbar.last_print_n = 0

        pbar = ProgressBar(persist=False)
        pbar.attach(train_engine)

        train_engine.add_event_handler(Events.ITERATION_COMPLETED(every=5000), compute_metrics)
        train_engine.add_event_handler(Events.EPOCH_COMPLETED, compute_metrics)

        early_stop_handler = EarlyStopping(
            patience=config_parameters['early_stop'],
            score_function=self._negative_loss,
            trainer=train_engine)
        
        inference_engine.add_event_handler(Events.EPOCH_COMPLETED, early_stop_handler)
        inference_engine.add_event_handler(Events.EPOCH_COMPLETED, checkpoint_handler, {'model': model,})

        train_engine.run(trainloader, max_epochs=config_parameters['epochs'])

        torch.save(model.state_dict(), os.path.join(outputdir,'model.pth'))

        return outputdir

if __name__ == "__main__":
    fire.Fire(Runner)