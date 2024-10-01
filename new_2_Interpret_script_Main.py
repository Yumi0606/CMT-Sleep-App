import os
import numpy as np
import matplotlib.pyplot as plt
import random
import mne
from mne.datasets.sleep_physionet.age import fetch_data
from datetime import date

import torch
from torchvision import transforms, datasets
from torch.utils import data
from torch.utils.data import Dataset, DataLoader

from pylab import mpl

import warnings

from einops import rearrange
import torch.nn.functional as F
import torch.nn as nn
import math

from matplotlib.collections import LineCollection

from matplotlib.colors import ListedColormap, BoundaryNorm

from models.sequence_cmt import Seq_Cross_Transformer_Network  # as Seq_Cross_Transformer_Network
from models.sequence_cmt import Epoch_Cross_Transformer
from models.model_blocks import PositionalEncoding, Window_Embedding, Intra_modal_atten, Cross_modal_atten, Feed_forward

import json

class SleepEDF_Seq_MultiChan_Dataset_Inference(Dataset):
    def __init__(self, eeg_file, eog_file, device, mean_eeg_l=None, sd_eeg_l=None,
                 mean_eog_l=None, sd_eog_l=None, mean_eeg2_l=None, sd_eeg2_l=None, transform=None,
                 target_transform=None, sub_wise_norm=False, num_seq=5):
        """

        """
        # Get the data

        self.eeg = eeg_file
        self.eog = eog_file
        # self.labels = label_file

        # self.labels = torch.from_numpy(self.labels)

        # bin_labels = np.bincount(self.labels)
        # print(f"Labels count: {bin_labels}")
        print(f"Shape of EEG : {self.eeg.shape} , EOG : {self.eog.shape}")  # , EMG: {self.eeg2.shape}")
        # print(f"Shape of Labels : {self.labels.shape}")

        if sub_wise_norm == True:
            print(f"Reading Subject wise mean and sd")

            self.mean_eeg = mean_eeg_l
            self.sd_eeg = sd_eeg_l
            self.mean_eog = mean_eog_l
            self.sd_eog = sd_eog_l

        self.sub_wise_norm = sub_wise_norm
        self.device = device
        self.transform = transform
        self.target_transform = target_transform
        self.num_seq = num_seq

    def __len__(self):
        return self.eeg.shape[0] - self.num_seq

    def __getitem__(self, idx):
        eeg_data = self.eeg[idx:idx + self.num_seq].squeeze()
        eog_data = self.eog[idx:idx + self.num_seq].squeeze()
        # label = self.labels[idx:idx + self.num_seq, ]

        if self.sub_wise_norm == True:
            eeg_data = (eeg_data - self.mean_eeg[idx]) / self.sd_eeg[idx]
            eog_data = (eog_data - self.mean_eog[idx]) / self.sd_eog[idx]
        elif self.mean and self.sd:
            eeg_data = (eeg_data - self.mean[0]) / self.sd[0]
            eog_data = (eog_data - self.mean[1]) / self.sd[1]
        if self.transform:
            eeg_data = self.transform(eeg_data)
            eog_data = self.transform(eog_data)
        # if self.target_transform:
        # label = self.target_transform(label)
        return eeg_data, eog_data

def plot_interpret(i, x, y, dydx, fig, axs, axs_no, signal_type="EEG"):
    points = np.array([x, y]).T.reshape(-1, 1, 2)
    segments = np.concatenate([points[:-1], points[1:]], axis=1)
    # plt.figure(figsize = (30,5))
    # plt.figure(figsize=(25,5))
    # plt.plot(x,dydx)
    # plt.title(f"Attention Map for Class {label}  {signal_type} ")
    # plt.xlim(x.min(),x.max())
    # plt.colorbar()

    # fig, axs = plt.subplots(2, 1, sharex=True, sharey=True,figsize = (30,10))

    # Create a continuous norm to map from data points to colors
    norm = plt.Normalize(dydx.min(), dydx.max())

    lc = LineCollection(segments, cmap='Reds', norm=norm)
    # Set the values used for colormapping
    lc.set_array(dydx)
    lc.set_linewidth(15)
    line = axs[axs_no[0]][axs_no[1]].add_collection(lc)
    # fig.colorbar(line, ax=axs[axs_no[0]][axs_no[1]])
    # fig.colorbar(line, ax=axs[1])
    # axs[axs_no[0]][axs_no[1]].set_xlabel(f"{signal_type}",fontsize = 100,labelpad = 20)
    axs[axs_no[0]][axs_no[1]].set_title(f'Epoch {i + 1} {signal_type}', fontsize=100)
    # axs[i].set_xlabel('Signal',fontsize = 100)
    # axs[axs_no[0]][axs_no[1]].axis('off')
    # Hide X and Y axes label marks
    axs[axs_no[0]][axs_no[1]].xaxis.set_tick_params(labelbottom=False)
    axs[axs_no[0]][axs_no[1]].yaxis.set_tick_params(labelleft=False)

    # Hide X and Y axes tick marks
    axs[axs_no[0]][axs_no[1]].set_xticks([])
    axs[axs_no[0]][axs_no[1]].set_yticks([])
    axs[axs_no[0]][axs_no[1]].set_xlim(x.min(), x.max())
    axs[axs_no[0]][axs_no[1]].set_ylim(y.min() - 0.2, y.max() + 0.2)


def atten_interpret(q, k):
    atten_weights = torch.softmax((q @ k.transpose(-2, -1) / math.sqrt(q.size(-1))), dim=-1)
    return atten_weights

def new_signal_extract_2(path_1, channel='eeg1', filter=True, freq=[0.2, 40]):
    all_channels = (
    'EEG Fpz-Cz', 'EEG Pz-Oz', 'EOG horizontal', 'EMG submental', 'Resp oro-nasal', 'Temp rectal', 'Event marker')

    data = [path_1]
    signal2idx = {"eeg1": 0, "eeg2": 1, "eog": 2, "emg": 3}

    all_channels_list = list(all_channels)
    all_channels_list.remove(all_channels[signal2idx[channel]])
    exclude_channels = tuple(all_channels_list)

    sleep_signals = mne.io.read_raw_edf(data[0], verbose=True, exclude=exclude_channels, preload=True)

    # Filtering
    tmax = 30. - 1. / sleep_signals.info['sfreq']

    if filter == True:
        sleep_signals = sleep_signals.copy().filter(l_freq=freq[0], h_freq=freq[1])

    duration = 30
    epochs = mne.make_fixed_length_epochs(sleep_signals, duration=duration, preload=True)

    # Calculate mean and std of the signal epochs
    signal_mean = np.mean(epochs)
    signal_std = np.std(epochs)

    main_ext_raw_data = epochs.get_data()
    main_sub_len = np.array([len(epochs)])
    main_mean = np.tile(signal_mean, (len(epochs), 1)).squeeze()
    main_std = np.tile(signal_std, (len(epochs), 1)).squeeze()

    return main_ext_raw_data, main_sub_len, main_mean, main_std

def main(file_name):
    # 创建picture文件夹保存图片
    current_directory = os.getcwd()
    new_folder_name = "static/picture"
    full_folder_path = os.path.join(current_directory, new_folder_name)
    os.makedirs(full_folder_path, exist_ok=True)

    # 设置显示中文字体
    mpl.rcParams["font.sans-serif"] = ["SimHei"]
    start_time_point = 38820  ### Set the time point in the signal to visualize.  ( For figure 6 in the paper Subject 38 Day 1 Start time point 38820)
    file_name = file_name

    path_1 = f'data/{file_name}'

    eeg1_1,  len_1, eeg1_m1, eeg1_std1 = new_signal_extract_2(path_1, channel='eeg1', filter=True, freq=[0.2, 40])
    eog_1, _, eog_m1, eog_std1 = new_signal_extract_2(path_1,  channel='eog', filter=True, freq=[0.2, 40])

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    num_seq = 15
    infer_dataset = SleepEDF_Seq_MultiChan_Dataset_Inference(eeg_file=eeg1_1,
                                                             eog_file=eog_1,
                                                             # label_file=labels_1,
                                                             device=device, mean_eeg_l=eeg1_m1, sd_eeg_l=eeg1_std1,
                                                             mean_eog_l=eog_m1, sd_eog_l=eog_std1,
                                                             sub_wise_norm=True, num_seq=num_seq, # wait for change
                                                             transform=transforms.Compose([
                                                                 transforms.ToTensor()
                                                             ]))

    infer_data_loader = data.DataLoader(infer_dataset, batch_size=1, shuffle=False)  # 16
    len(infer_data_loader)

    eeg_data, eog_data= next(iter(infer_data_loader))
    # print(f"EEG batch shape: {eeg_data.size()}")
    # print(f"EOG batch shape: {eog_data.size()}")
    # print(f"EMG batch shape: {eeg2_data.size()}")
    # print(f"Labels batch shape: {label.size()}")

    eeg_data_temp = eeg_data[0].squeeze()  # (0)
    eog_data_temp = eog_data[0].squeeze()  # (0)

    # print(eeg_data_temp.shape)

    t = np.arange(0, 30, 1 / 100)
    plt.figure(figsize=(10, 5))
    plt.plot(eeg_data_temp[0].squeeze())
    plt.plot(eog_data_temp[0].squeeze() + 5)
    plt.title(f"EEG & EOG表格")

    save_dir = "static/picture"
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    save_path = os.path.join(save_dir, "picture_1.jpg")
    plt.savefig(save_path, dpi=300)

    # plt.show()# show放在savefig前会输出空白图片,这里修改了

    test_model = torch.load('./checkpoint_model_best_acc.pth.tar', map_location=device)
    test_model.eval()
    # print(sum(p.numel() for p in test_model.parameters() if p.requires_grad))

    """# Get Predictions for the Subject"""

    warnings.filterwarnings("ignore")
    '''
    # feat_main = []
    pred_val_main = torch.zeros((len(infer_data_loader) + num_seq, 1, 5))  # data, output,seq pred,
    # labels_val_main = torch.zeros((len(infer_data_loader) + num_seq, 1))  # data, output,seq pred,
    m = torch.nn.Softmax()
    with torch.no_grad():  # 确保在接下来的代码块中不会计算梯度
        test_model.eval()  # 将模型设置为评估模式，这对于推断是必要的
        for batch_val_idx, data_val in enumerate(infer_data_loader):  # 遍历数据加载器中的每个批次
            if batch_val_idx % 1 == 0: print("predicting", batch_val_idx)
            val_eeg, val_eog = data_val  # 从批次数据中解包EEG、EOG信号和标签
            pred, _ = test_model(val_eeg.float().to(device), val_eog.float().to(device))  # 使用模型进行预测，忽略返回的第二个值
            
            print("#########")
            print("Start")
            print(pred)
            print("End")
            print("#########")
            

            # feat_main.append(feat_list)  # 这行代码被注释掉了，它看起来像是用来存储特征的
            for ep in range(num_seq):  # 遍历每个序列
                pred_val_main[batch_val_idx + ep] += m(pred[ep]).cpu()  # 将预测结果累加到pred_val_main数组中，m可能是一个映射函数

    # print(pred_val_main[0],pred_val_main[1000])

    # pred_val_main = (pred_val_main / num_seq).squeeze()  # 计算预测的平均值
    '''
    """## Get Interpretations for the Subject"""

    infer_data_loader = data.DataLoader(infer_dataset, batch_size=1, shuffle=False)  # 16
    batch_size = len(infer_data_loader)
    infer_data_loader = data.DataLoader(infer_dataset, batch_size=batch_size, shuffle=False)  # 16

    t = 1450

    eeg_data, eog_data = next(iter(infer_data_loader))

    l = eeg_data.shape[0] // 15

    new_pred = []

    for i in range(l):
        pred, feat_list = test_model(eeg_data[i*15].unsqueeze(0).float().to(device), eog_data[i*15].unsqueeze(0).float().to(device))
        pred = np.array([i.argmax(-1).item() for i in pred])
        for j in range(num_seq):
            new_pred.append(pred[j])

    pred, feat_list = test_model(eeg_data[t].unsqueeze(0).float().to(device), eog_data[t].unsqueeze(0).float().to(device))
    pred = np.array([i.argmax(-1).item() for i in pred])


    for i in feat_list[:-1]:
        print(i[0].shape, i[1].shape, i[2].shape)

    # print(feat_list[-1].shape)

    label_dict = ['Wake', 'N1', 'N2', 'N3', 'REM']
    pred_list = [label_dict[i] for i in pred]
    print("pred_list",pred_list)
    print("new_pred_list",new_pred)

    data_new = [
        {'name': 'N1期', 'value': 0},
        {'name': 'N2期', 'value': 0},
        {'name': 'N3期', 'value': 0},
        {'name': 'REM期', 'value': 0}
    ]

    for i in new_pred:
        if i == 1:
            data_new[0]['value'] += 0.5
        elif i == 2:
            data_new[1]['value'] += 0.5
        elif i == 3:
            data_new[2]['value'] += 0.5
        elif i == 4:
            data_new[3]['value'] += 0.5

    # 定义文件夹和文件名
    directory = "sleep_proportion"  # 存储数据的文件夹名称
    file_name = "sleep_data.json"  # 存储数据的文件名称
    file_path = os.path.join(directory, file_name)  # 拼接完整的文件路径

    # 如果文件夹不存在，则创建文件夹
    if not os.path.exists(directory):
        os.makedirs(directory)  # 创建文件夹

    # 将data_new数据保存到JSON文件中
    with open(file_path, 'w') as file:  # 打开文件用于写入
        json.dump(data_new, file)  # 将data_new转换为JSON格式并写入文件


    ## 画图 ##
    ###### Interpreting inter-epoch relationships  ##########
    plt.rcParams['axes.linewidth'] = 2
    seq_features = feat_list[-1]  ##extracting learned inter-epoch features
    # seq_atten = atten_interpret(seq_features.squeeze(),seq_features.squeeze()).squeeze().detach().cpu().numpy()
    # print(seq_atten.shape)
    # plt.figure()
    # plt.imshow(seq_atten)

    fig, axs = plt.subplots(15, 1, figsize=(1 * 5, 15 * 8))
    seq_atten_list = []
    for i in range(num_seq):
        seq_atten = atten_interpret(seq_features.squeeze()[i].unsqueeze(0),
                                    seq_features.squeeze()).squeeze().detach().cpu().numpy()

        rgba_colors = np.zeros((num_seq, 4))
        rgba_colors[:, 0] = 0  # value of red intensity divided by 256
        rgba_colors[i, 0] = 0.4  # value of red intensity divided by 256
        rgba_colors[:, 1] = 0  # value of green intensity divided by 256
        rgba_colors[:, 2] = 0.4  # value of blue intensity divided by 256
        rgba_colors[i, 2] = 0
        seq_atten = seq_atten / seq_atten.max()

        seq_atten_list.append(seq_atten)  #
        rgba_colors[:, -1] = seq_atten
        axs[i].bar(np.arange(1, 16), seq_atten / seq_atten.max(),  # color ='blue',
                   color=rgba_colors, align='center', width=0.8)
        # axs[i//5][i%5].set_title('')
        axs[i].tick_params(axis='x', labelsize=30)  # ,which = 'both')
        axs[i].tick_params(axis='y', labelsize=30)
        axs[i].set_xlabel('Epochs', fontsize=30)
        yticks = axs[i].yaxis.get_major_ticks()
        yticks[0].label1.set_visible(False)

    save_dir = "static/picture"
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    save_path = os.path.join(save_dir, "picture_2.jpg")
    plt.savefig(save_path, dpi=300)

    ###### Interpreting cross-modal relationships  ##########
    fig, axs = plt.subplots(15, 1, figsize=(1 * 5, 15 * 10))

    cross_atten_list = []  #
    from matplotlib.font_manager import FontProperties
    my_font = FontProperties(fname='env/simhei.ttf')

    for i in range(num_seq):
        cross_features = feat_list[i][-1]  ##extracting learned cross-modal features
        cross_atten = atten_interpret(cross_features.squeeze()[0].unsqueeze(0),
                                      cross_features.squeeze()[1:]).squeeze().detach().cpu().numpy()
        cross_atten_list.append(cross_atten)  #

        rgba_colors = np.zeros((2, 4))
        rgba_colors[:, 0] = 0.4  # value of red intensity divided by 256
        rgba_colors[:, 1] = 0  # value of green intensity divided by 256
        rgba_colors[:, 2] = 0  # value of blue intensity divided by 256
        rgba_colors[:, -1] = cross_atten + 0.1
        axs[i].bar(['EEG', 'EOG'], cross_atten,  # color ='red',
                   color=rgba_colors, align='center', width=0.9)
        axs[i].tick_params(axis='x', labelsize=30)  # ,which = 'both')
        axs[i].tick_params(axis='y', labelsize=30)
        axs[i].set_ylim(0, 1.02)
        axs[i].set_xlabel('注意力占比', fontsize=30, fontproperties=my_font)
    # 创建文件名，包含当前日期时间

    save_dir = "static/picture"
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    save_path = os.path.join(save_dir, "picture_3.jpg")
    plt.savefig(save_path, dpi=300)
    # plt.savefig(f'/content/cross_modal_sub_{subject_no}_day_{days}_t_{t}_part_1.pdf',dpi = 300)

    ###### Interpreting intra-modal relationships  ##########
        ###### 解释同模态内部关系  ##########
    plt.rcParams['axes.linewidth'] = 20
    fig, axs = plt.subplots(15, 2, figsize=(2 * 50, 15 * 20))
    # seq_features = feat_list[-1]
    eeg_atten_list = []  #
    eog_atten_list = []  #
    for i in range(num_seq):
        eeg_features = feat_list[i][0]  ##extracting learned intra-modal EEG features
        eog_features = feat_list[i][1]  ##extracting learned intra-modal EOG features
        cross_features = feat_list[i][-1]  ##extracting learned cross-modal features

        eeg_atten = atten_interpret(cross_features.squeeze()[0].unsqueeze(0),
                                    eeg_features.squeeze()[1:])  # .squeeze().detach().cpu().numpy()
        eog_atten = atten_interpret(cross_features.squeeze()[0].unsqueeze(0),
                                    eog_features.squeeze()[1:])  # .squeeze().detach().cpu().numpy()

        eeg_atten = F.upsample(eeg_atten.unsqueeze(0), scale_factor=3000 // 60,
                               mode='nearest').squeeze().detach().cpu().numpy()
        eog_atten = F.upsample(eog_atten.unsqueeze(0), scale_factor=3000 // 60,
                               mode='nearest').squeeze().detach().cpu().numpy()

        eeg_atten_list.append(eeg_atten)  #
        eog_atten_list.append(eog_atten)  #

        t1 = np.arange(0, 30, 1 / 100)
        plot_interpret(i, t1, eeg_data[t, 0, i, :].squeeze().cpu().numpy(), eeg_atten, fig, axs, [i, 0], signal_type="EEG")
        plot_interpret(i, t1, eog_data[t, 0, i, :].squeeze().cpu().numpy(), eog_atten, fig, axs, [i, 1], signal_type="EOG")

    save_dir = "static/picture"
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    save_path = os.path.join(save_dir, "picture_4.jpg")
    plt.savefig(save_path, dpi=200)



    """# Final plot similar to the paper"""
    # fig, axs = plt.subplots(num_seq, 4,figsize=(200, 20*num_seq),gridspec_kw={'width_ratios': [2,2,10,10]}) # for more clear figure
    fig, axs = plt.subplots(num_seq, 4, figsize=(100, 10 * num_seq), gridspec_kw={'width_ratios': [2, 2, 10, 10]})
    title_font_size = fig.dpi * 0.4
    label_font_size = fig.dpi * 0.2
    for i in range(num_seq):
        # Plotting inter-epoch attention ##############################
        rgba_colors = np.zeros((num_seq, 4))
        rgba_colors[:, 0] = 0  # value of red intensity divided by 256
        rgba_colors[i, 0] = 0.4  # value of red intensity divided by 256
        rgba_colors[:, 1] = 0  # value of green intensity divided by 256
        rgba_colors[:, 2] = 0.4  # value of blue intensity divided by 256
        rgba_colors[i, 2] = 0
        rgba_colors[:, -1] = seq_atten_list[i]
        axs[i][0].bar(np.arange(1, num_seq + 1), seq_atten_list[i] / seq_atten_list[i].max(),
                      # /seq_attn[i].max(),# color ='blue',
                      color=rgba_colors, align='center')
        # axs[i//5][i%5].set_title('')
        axs[i][0].tick_params(axis='x', labelsize=label_font_size)
        axs[i][0].tick_params(axis='y', labelsize=label_font_size)
        axs[i][0].set_xlabel('Epochs', fontsize=title_font_size)
        yticks = axs[i][0].yaxis.get_major_ticks()
        yticks[0].label1.set_visible(False)

        # Plotting cross-modal attention ##############################
        rgba_colors = np.zeros((2, 4))
        rgba_colors[:, 0] = 0.4  # value of red intensity divided by 256
        rgba_colors[:, 1] = 0  # value of green intensity divided by 256
        rgba_colors[:, 2] = 0  # value of blue intensity divided by 256
        rgba_colors[:, -1] = cross_atten_list[i]
        axs[i][1].bar(['EEG', 'EOG'], cross_atten_list[i],  # color ='red',
                      color=rgba_colors, align='center')
        axs[i][1].tick_params(axis='x', labelsize=label_font_size)
        axs[i][1].tick_params(axis='y', labelsize=label_font_size)
        axs[i][1].set_ylim(0, 1.02)
        axs[i][1].set_xlabel('Signal', fontsize=title_font_size)

        # # Plotting EEG attention ##############################
        eeg_atten_epoch = eeg_atten_list[i]
        t1 = np.arange(0, 3000, 1)
        plot_interpret(i, t1, eeg_data[t, 0, i, :].squeeze().cpu().numpy(), eeg_atten, fig, axs, [i, 2],
                       signal_type=f"EEG Class:{pred_list[i]}")

        # plot_interpret(t1,eog_data[t,0,i,:].squeeze().cpu().numpy(),eog_atten,fig,[i,1],signal_type = "EOG")

        # # Plotting EOG attention #
        eog_atten_epoch = eog_atten[i]
        plot_interpret(i, t1, eog_data[t, 0, i, :].squeeze().cpu().numpy(), eog_atten, fig, axs, [i, 3],
                       signal_type=f"EOG Class:{pred_list[i]}")

    # time = [int(record_id.split('-')[1].split('_')[i]) for i in range(num_epoch_seq)]
    # plt.subplots_adjust(wspace=0.2)
    # fig.suptitle('Interpretation for patient '+str([38])+' for 30s epochs from '+str(start_time_point)+'s',fontsize = title_font_size*2)

    save_dir = "static/picture"
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    save_path = os.path.join(save_dir, "picture_5.jpg")
    plt.savefig(save_path, dpi=100)

if __name__ == '__main__':
    main()
