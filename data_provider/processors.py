import os
import joblib
import numpy as np
import pandas as pd
from glob import glob
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import pickle
import numpy as np
import torch

class BasePreprocessor:
    def __init__(self, args):
        self.root_path = args.raw_root_path
        self.save_path = args.sta_root_path
        self.test_size = args.test_size
        self.args = args
        os.makedirs(self.root_path, exist_ok=True)
        os.makedirs(self.save_path, exist_ok=True)

    def load_and_clean_data(self):
        raise NotImplementedError

    def run(self):
        X, y = self.load_and_clean_data()

        # 生成全局唯一ID
        ids = np.arange(len(X))

        # split 时带上 ids
        X_train, X_test, y_train, y_test, id_train, id_test = train_test_split(
            X, y, ids,
            test_size=self.test_size,
            shuffle=True,
            random_state=42,
            stratify=y
        )

        # 保存归一化信息（原逻辑）
        max_feature = np.max(X_train, axis=(0, 1))
        max_feature = np.where(max_feature == 0, 1.0, max_feature)
        np.save(os.path.join(self.save_path, 'max_feature.npy'), max_feature)

        N, T, F = X_train.shape
        scaler = StandardScaler().fit(X_train.reshape(-1, F))
        joblib.dump(scaler, os.path.join(self.save_path, 'scaler.pkl'))

        if self.args.do_stasca:
            X_train = scaler.transform(X_train.reshape(-1, F)).reshape(-1, T, F)
            X_test = scaler.transform(X_test.reshape(-1, F)).reshape(-1, T, F)

        # ✅ 保存数据 + ID
        np.save(os.path.join(self.save_path, 'X_train.npy'), X_train)
        np.save(os.path.join(self.save_path, 'y_train.npy'), y_train)
        np.save(os.path.join(self.save_path, 'id_train.npy'), id_train)

        np.save(os.path.join(self.save_path, 'X_test.npy'), X_test)
        np.save(os.path.join(self.save_path, 'y_test.npy'), y_test)
        np.save(os.path.join(self.save_path, 'id_test.npy'), id_test)

        print("✅ Data + IDs saved!")

# ==========================================
# 具体实现类 (Ali)
# ==========================================
class AliPreprocessor(BasePreprocessor):
    def load_and_clean_data(self):

        data_path = [os.path.join(self.root_path, dp) for dp in self.args.data_path]

        X = []
        y = []
        for file in data_path:
            ins = joblib.load(file)
            data = np.stack([sample['data'].values for sample in ins])
            label = np.array([sample['label'] for sample in ins])
            label = (label != 0).astype(int) # 二分类
            X.append(data)
            y.append(label)

        X = np.concatenate(X, axis=0)
        y = np.concatenate(y, axis=0)

        fault_ins = np.where(y > 0)[0]
        health_ins = np.where(y == 0)[0]

        fault_X = X[fault_ins]
        fault_y = y[fault_ins]
        health_X = X[health_ins]
        health_y = y[health_ins]
        
        fault_len = len(fault_X)
        health_len = fault_len * 200
        if health_len > len(health_X):
            health_len = len(health_X)
        health_X = health_X[:health_len]
        health_y = health_y[:health_len]
        X = np.concatenate([health_X, fault_X], axis=0)
        y = np.concatenate([health_y, fault_y], axis=0)

        return X, y