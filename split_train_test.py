import os
import pandas as pd
import shutil
from sklearn.model_selection import train_test_split

# Paths
train_csv_path = 'train.csv'
train_folder = 'dataset/train'
test_folder = 'dataset/test'
test_csv_path = 'test.csv'

# Create test folder if it doesn't exist
os.makedirs(test_folder, exist_ok=True)

# Read the train.csv file
df = pd.read_csv(train_csv_path)

# Split the data into train and test sets (90% train, 10% test)
train_df, test_df = train_test_split(df, test_size=0.1, random_state=42)
# Save the updated train set to train.csv
train_df.to_csv(train_csv_path, index=False)
# Save the test set to test.csv
test_df.to_csv(test_csv_path, index=False)

# Move the test images to the test folder
for idx in test_df['Id']:
    src_path = os.path.join(train_folder, f'{idx}.jpg')
    dst_path = os.path.join(test_folder, f'{idx}.jpg')
    if os.path.exists(src_path):
        shutil.move(src_path, dst_path)