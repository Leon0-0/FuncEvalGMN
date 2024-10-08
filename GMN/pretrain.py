from GMN.handle_dataset.spider_dataset import SpiderDataset
# from GMN.handle_dataset.data_pre_processor.ast_processor import ASTProcessor
from GMN.handle_dataset.data_pre_processor.positional_encoding_processor import PositionalEncodingProcessor
from utils import *
from GMN.configure import *
import numpy as np
import os
import random
import pandas as pd
from scipy.ndimage import gaussian_filter1d
import matplotlib.pyplot as plt
from torch_geometric.data import Batch


class DataLoader(torch.utils.data.DataLoader):
    r"""Data loader which merges data objects from a
    :class:`torch_geometric.data.dataset` to a mini-batch.

    Args:
        dataset (Dataset): The dataset from which to load the data.
        batch_size (int, optional): How may samples per batch to load.
            (default: :obj:`1`)
        shuffle (bool, optional): If set to :obj:`True`, the data will be
            reshuffled at every epoch (default: :obj:`True`)
    """

    def __init__(self, dataset, batch_size=1, shuffle=True, **kwargs):
        super(DataLoader, self).__init__(
            dataset,
            batch_size,
            shuffle,
            collate_fn=lambda data_list: Batch.from_data_list(data_list),
            **kwargs)

seed = 8
os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'
np.random.seed(seed)
random.seed(seed)
torch.cuda.manual_seed(seed)
torch.manual_seed(seed)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
torch.cuda.manual_seed_all(seed)
torch.use_deterministic_algorithms(True)

is_cuda = torch.cuda.is_available()
ONLINE_TRAIN_SETTINGS = {
    'NAME': "pretrain_new",
    'GPU_DEVICE': torch.device('cuda:0' if is_cuda else 'cpu'),
    'PATH_TO_GMN': "./GMN",
    'IMAGES_FOLDER': "save_file/picture/",
    'LOG_FILE_NAME': "save_file/log/plot",
    'Checkpoint': 'save_file/checkpoints/',
    'Train': './GMN/database_spider/gmn_api_data/Spider_pair_train.xlsx',
    'Dev': './GMN/database_spider/gmn_api_data/Spider_pair_dev.xlsx',
}

print("==================================")
device = ONLINE_TRAIN_SETTINGS['GPU_DEVICE']
name = ONLINE_TRAIN_SETTINGS['NAME']
print("Device:", device)
print("Name:", name)

if is_cuda:
    image_folder_path = os.path.join(ONLINE_TRAIN_SETTINGS['PATH_TO_GMN'], ONLINE_TRAIN_SETTINGS['IMAGES_FOLDER'])
    log_folder_path = os.path.join(ONLINE_TRAIN_SETTINGS['PATH_TO_GMN'], ONLINE_TRAIN_SETTINGS['LOG_FILE_NAME'])
    checkpoint_folder_path = os.path.join(ONLINE_TRAIN_SETTINGS['PATH_TO_GMN'], ONLINE_TRAIN_SETTINGS['Checkpoint'])

    train_path = os.path.join(ONLINE_TRAIN_SETTINGS['PATH_TO_GMN'], ONLINE_TRAIN_SETTINGS['Train'])
    dev_path = os.path.join(ONLINE_TRAIN_SETTINGS['PATH_TO_GMN'], ONLINE_TRAIN_SETTINGS['Dev'])

    print("Images saved in:", image_folder_path)
    print("Logs saved in:", log_folder_path)
    print("==================================")

else:
    image_folder_path = ONLINE_TRAIN_SETTINGS['IMAGES_FOLDER']
    log_folder_path = ONLINE_TRAIN_SETTINGS['LOG_FILE_NAME']
    checkpoint_folder_path = ONLINE_TRAIN_SETTINGS['Checkpoint']

    train_path = ONLINE_TRAIN_SETTINGS['Train']
    dev_path = ONLINE_TRAIN_SETTINGS['Dev']


torch.set_default_tensor_type(torch.FloatTensor)
# Print configure
config = get_CL_config()
for (k, v) in config.items():
    print("%s= %s" % (k, v))

train_df = pd.read_excel(train_path, sheet_name="Sheet1").reset_index(drop=True)
dev_df = pd.read_excel(dev_path, sheet_name="Sheet1").reset_index(drop=True)

aug1 = 'dropN'
ratio1 = 0.1
aug2 = 'maskN'
ratio2 = 0.1

data_pre_processor = PositionalEncodingProcessor()

log_txt = ""

model, optimizer = build_model(config)
model.to(device)

loss_values = []
def split_data_randomly(df, split_ratio=0.5):
    # 打乱数据
    df_shuffled = df.sample(frac=1, random_state=42).reset_index(drop=True)
    # 划分数据
    split_index = int(len(df_shuffled) * split_ratio)
    df_part1 = df_shuffled.iloc[:split_index]
    df_part2 = df_shuffled.iloc[split_index:]
    df_part2 = df_part2.reset_index(drop=True)
    return df_part1, df_part2

model.train(True)

for i_epoch in range(201):

    train_df = train_df.sample(frac=1, random_state=42).reset_index(drop=True)
    
    pair_list_train1, labels_train = data_pre_processor.read_data_cl(train_df, aug1, ratio1)
    train_batch_data1 = data_pre_processor.pairs_spider(config['training']['batch_size'], pair_list_train1, labels_train)

    pair_list_train2, labels_train = data_pre_processor.read_data_cl(train_df, aug2, ratio2)
    train_batch_data2 = data_pre_processor.pairs_spider(config['training']['batch_size'], pair_list_train2, labels_train)

    training_dataset1 = SpiderDataset(train_batch_data1)
    training_dataset2 = SpiderDataset(train_batch_data2)

    for (edge_tuple1, node_tuple1, n_graphs1, batch_labels1), (edge_tuple2, node_tuple2, n_graphs2, batch_labels2) in zip(training_dataset1, training_dataset2):
        optimizer.zero_grad()
        out1 = model.forward_cl(edge_tuple1.to(device), node_tuple1.to(device), n_graphs1)
        out2 = model.forward_cl(edge_tuple2.to(device), node_tuple2.to(device), n_graphs2)
        loss = model.loss_cl(out1, out2)
        loss_values.append(loss.item()) 
        loss.backward()
        print(loss)
        optimizer.step()

    y_smoothed = gaussian_filter1d(loss_values, sigma=5)
    plt.clf()
    plt.title('Training Loss')
    plt.xlabel('iterations')
    plt.ylabel('Loss')
    plt.plot( loss_values, label='Original')
    plt.plot(y_smoothed, label='Smooth')
    plt.grid(True)
    plt.legend()
    plt.savefig('./GMN/pretrain_loss/pretrain_' + str(aug1) + '_' + str(ratio1) + '_' + str(aug2) + '_' + str(ratio2) + '_' + '1e5' + '.png')

    if i_epoch % 25 == 0 and i_epoch != 0:
        torch.save(model.state_dict(), './GMN/models/pretrain_model_' + str(ratio1) + '_' + str(aug1) + '_'+ str(ratio2) + '_' + str(aug2) + '_' + str(i_epoch) + '_' + '1e5' + '.pt')
        







