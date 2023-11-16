'''
Here we have all functions to visualize results.
'''

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sn
from numpy import random
from sklearn.metrics import confusion_matrix


def ConfusionMatrix(out_path, trues, preds, labels, threshold, bit_resolution=8, use_trainable_tc=False, use_trainable_out=False, repetition=1):
    '''
    Here we create a confusion matrix.
    '''

    cm = confusion_matrix(y_true=trues, y_pred=preds, labels=labels, normalize='true')
    cm_df = pd.DataFrame(cm, index=[ii for ii in labels], columns=[
                         jj for jj in labels])
    plt.figure(figsize=(12, 12))
    sn.heatmap(cm_df,
               annot=True,
               fmt='.1g',
               cbar=False,
               square=False,
               cmap="YlGnBu")
    plt.xlabel('\nPredicted')
    plt.ylabel('True\n')
    plt.xticks(rotation=0)

    local_path = f'rsnn_thr_{threshold}'
    if use_trainable_tc:
        local_path += '_train_tc'
    if use_trainable_out:
        local_path += '_train_out'
    plt.savefig(
        f"{out_path}/{local_path}_{bit_resolution}_bit_resolution_confusion_matrix_run_{repetition+1}.png", dpi=300)


def NetworkActivity(out_path, spk_recs, threshold, bit_resolution=8, use_trainable_tc=False, use_trainable_out=False, repetition=1):
    '''
    Here we visualize the network activity of a random batch.
    '''

    # select a random sample from the batch
    sample_num = random.randint(0, spk_recs[0].shape[0])
    # nb_plt = len(spk_recs)
    # gs = GridSpec(1, nb_plt)
    for layer_num, layer_activity in enumerate(spk_recs):
        # plot layer activity as rasterplot
        plt.figure(figsize=(16, 9))
        plt.title(f"Layer {layer_num}")
        # for i in range(nb_plt):
        # plt.subplot(gs[layer_num])
        plt.imshow(layer_activity[sample_num].T,
                   cmap=plt.cm.gray_r, origin="lower", aspect='auto')
        # if i == 0:
        plt.xlabel("Time")
        plt.ylabel("Units")
        sn.despine()
        plt.tight_layout()
        local_path = f'rsnn_thr_{threshold}_layer_{layer_num+1}'
        if use_trainable_tc:
            local_path += '_train_tc'
        if use_trainable_out:
            local_path += '_train_out'
        plt.savefig(
            f"{out_path}/{local_path}_{bit_resolution}_bit_resolution_raster_plot_run_{repetition+1}.png", dpi=300)
