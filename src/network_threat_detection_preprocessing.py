import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

csv_files = [
    '../datasets/Friday-WorkingHours-Afternoon-DDos.pcap_ISCX.csv',
    '../datasets/Friday-WorkingHours-Afternoon-PortScan.pcap_ISCX.csv',
    '../datasets/Friday-WorkingHours-Morning.pcap_ISCX.csv',
    '../datasets/Monday-WorkingHours.pcap_ISCX.csv',
    '../datasets/Thursday-WorkingHours-Afternoon-Infilteration.pcap_ISCX.csv',
    '../datasets/Thursday-WorkingHours-Morning-WebAttacks.pcap_ISCX.csv',
    '../datasets/Tuesday-WorkingHours.pcap_ISCX.csv',
    '../datasets/Wednesday-workingHours.pcap_ISCX.csv'
]

for file in csv_files:
    data = pd.read_csv(file)
    columns_to_drop = [' Fwd Header Length.1', 'Fwd Avg Bytes/Bulk', ' Fwd Avg Packets/Bulk', ' Fwd Avg Bulk Rate', ' Bwd Avg Bytes/Bulk', ' Bwd Avg Packets/Bulk', 'Bwd Avg Bulk Rate']
    data = data.drop(columns=columns_to_drop, errors='ignore')

    label_encoder = LabelEncoder()

    X = pd.DataFrame()
    y = pd.DataFrame()

    if ' Label' in data.columns:
        print("Column 'Label' exists in the DataFrame.")
        X = data.drop(columns=[' Label'])
        y = data[' Label']
    else:
        print("Column 'Label' does not exist in the DataFrame.")
        exit()

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    scaler = StandardScaler()

    # Replace positive infinity values with NaN
    X_train.replace([np.inf, -np.inf], np.nan, inplace=True)

    # Fill NaN values with 0
    X_train.fillna(0, inplace=True)

    # Fit the scaler to X_train and transform X_train
    X_train = scaler.fit_transform(X_train)

    # Replace positive infinity values with NaN in X_test
    X_test.replace([np.inf, -np.inf], np.nan, inplace=True)

    # Fill NaN values with 0 in X_test
    X_test.fillna(0, inplace=True)

    # Now you can apply the scaler to X_test
    X_test = scaler.transform(X_test)