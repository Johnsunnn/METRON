import joblib
import numpy as np
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from torch.utils.data import Dataset, DataLoader, Subset
import pandas as pd
import torch
import configs


class MetabonomicsDataset(Dataset):
    def __init__(self, data, targets, train=True):
        self.train = train
        self.data = data
        self.targets = targets

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        sample = self.data[idx]
        label = self.targets[idx]
        return sample, label

    def get_targets(self):
        return self.targets


def get_ba_dataset(batch_size):
    data = pd.read_csv('./dataset/BA_Modeling_all_samples_raw_data.csv')
    features = data.drop(columns=['Id', 'Rep', 'Habit', 'Batch', 'CA', 'Group'])
    labels = data['CA']
    groups = data['Group']

    categorical_features = features[['Sex', 'BT_ABO', 'BT_Rh', 'Ethnicity']]
    numerical_features = features.drop(columns=['Sex', 'BT_ABO', 'BT_Rh', 'Ethnicity'])
    numerical_cols = numerical_features.columns.tolist()

    features_categorical_onehot = pd.get_dummies(categorical_features, dtype=float)

    numerical_scaler = StandardScaler()
    features_numerical_scaled = numerical_scaler.fit_transform(numerical_features)
    features_numerical_scaled = pd.DataFrame(features_numerical_scaled, columns=numerical_cols)

    label_scaler = MinMaxScaler()

    scaled_labels = label_scaler.fit_transform(labels.values.reshape(-1, 1))
    scaled_labels = pd.DataFrame(scaled_labels, columns=['CA'])

    joblib.dump(label_scaler, './dataset/label_scaler.pkl')

    final_features = pd.concat([features_categorical_onehot, features_numerical_scaled, scaled_labels, groups], axis=1)

    train_data = final_features[final_features['Group'] == 'Modeling']
    test_data = final_features[final_features['Group'] == 'Validation']
    x_train = train_data.drop(columns=['Group', 'CA'])
    y_train = train_data['CA']
    x_test = test_data.drop(columns=['Group', 'CA'])
    y_test = test_data['CA']

    x_train = torch.tensor(x_train.values, dtype=torch.float32)
    y_train = torch.tensor(y_train.values, dtype=torch.float32)
    x_test = torch.tensor(x_test.values, dtype=torch.float32)
    y_test = torch.tensor(y_test.values, dtype=torch.float32)

    train_dataset = MetabonomicsDataset(x_train, y_train, train=True)
    test_dataset = MetabonomicsDataset(x_test, y_test, train=False)

    train_dataloader = DataLoader(dataset=train_dataset,
                                  batch_size=batch_size,
                                  shuffle=True,
                                  num_workers=0,
                                  pin_memory=True if configs.DEVICE.type == 'cuda' else False)

    test_dataloader = DataLoader(dataset=test_dataset,
                                 batch_size=batch_size,
                                 shuffle=False,
                                 num_workers=0,
                                 pin_memory=True if configs.DEVICE.type == 'cuda' else False)

    return train_dataloader, test_dataloader