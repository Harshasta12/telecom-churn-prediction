import kagglehub
import os

# Download latest version
path = kagglehub.dataset_download("keyush06/telecom-churncsv")

print("Path to dataset files:", path)


path = kagglehub.dataset_download("becksddf/churn-in-telecoms-dataset")  # or whatever the dataset slug was
print(path)



folder = r'C:\Users\harsh\.cache\kagglehub\datasets\becksddf\churn-in-telecoms-dataset\versions\1'
print(os.listdir(folder))