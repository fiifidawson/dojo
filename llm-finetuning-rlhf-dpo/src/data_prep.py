import pandas as pd
import numpy as np
from datasets import DatasetDict, Dataset
from huggingface_hub import login
from prompt.data_prep_prompt import template
from dotenv import load_dotenv

load_dotenv()

login(token="HF_TOKEN")

# Load data
df = pd.read_csv('data/idea-title_pairs-preferences.csv')

print("=" * 80)
print(df.head())
print("=" * 80)

def idea_to_prompt(idea):
    return [{"role": "user", "content": template(idea.lower())}]

df['prompt'] = df['idea'].apply(idea_to_prompt)

# Create chosen and rejected responses
def title_to_completion(title):
    return [{"role": "assistant", "content": title}]




# For a single Dataset object
# dataset.push_to_hub("your_username/your_dataset_name")

# For a DatasetDict (e.g., with 'train' and 'test' splits)
# dataset is the variable holding the DatasetDict
# dataset.push_to_hub("your_username/your_dataset_name")

# =======================================================
# Create a new dataset repository

# from huggingface_hub import HfApi
# api = HfApi()
# repo_id = "your_username/your_dataset_name"
# api.create_repo(repo_id=repo_id, repo_type="dataset", private=False) # Set private=True for a private dataset

# Upload a single file
# api.upload_file(
#     path_or_fileobj="/path/to/local/file.csv",
#     path_in_repo="data/file.csv", # Desired path within the repo
#     repo_id=repo_id,
#     repo_type="dataset",
# )

# Upload an entire folder:
# api.upload_folder(
#     folder_path="/path/to/local/folder",
#     repo_id=repo_id,
#     repo_type="dataset",
#     # Optional: use allow_patterns or ignore_patterns to filter files
# )


