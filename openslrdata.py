import os
from datasets import Dataset

# Path to the root of the dataset
root_path = "./openslr/"

def load_dataset(root_path):
    # Read the TSV file and extract data
    tsv_file_path = os.path.join(root_path, "data.tsv")
    with open(tsv_file_path, "r", encoding="utf-8") as tsv_file:
        lines = tsv_file.readlines()

    # Prepare data for the Hugging Face dataset
    file_paths = []
    folder_names = []
    texts = []

    for line in lines:
        file_name, _, text = line.strip().split("\t")
        folder_name = file_name[:2]

        file_path = os.path.join(root_path, "data", folder_name, file_name + ".flac")

        file_paths.append(file_path)
        folder_names.append(folder_name)
        texts.append(text)

    # Create the Hugging Face dataset
    dataset_dict = {
        "file_path": file_paths,
        "folder_name": folder_names,
        "text": texts,
    }

    # Wrap the dictionary in a Hugging Face Dataset object
    dataset = Dataset.from_dict(dataset_dict)

    from datasets import Dataset, Audio

    # Assuming you already have the lists of file paths, folder names, and texts from your TSV file
    dataset_dict = {
        "file_path": file_paths,
        "folder_name": folder_names,
        "text": texts,
    }

    # Wrap the dictionary in a Hugging Face Dataset object
    dataset = Dataset.from_dict(dataset_dict)

    # Cast the 'file_path' column to the 'Audio' feature
    dataset = dataset.cast_column("file_path", Audio())

    return dataset