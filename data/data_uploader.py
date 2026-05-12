from clearml import Dataset, Task
import os
from datasets import load_dataset
from sklearn.model_selection import train_test_split
import pandas as pd

task = Task.init(
    project_name="Small Group Project - Team 2",
    task_name="Upload PlantVillage Dataset",
    task_type=Task.TaskTypes.data_processing
)

task.execute_remotely(queue_name="default")


SEED = 100

## 1. Load the Plant Village dataset from the Hugging Face Hub 
plant_village = load_dataset("DScomp380/plant_village")['train']
dataset = pd.DataFrame(plant_village)
class_names = plant_village.features['label'].names
dataset = dataset[dataset['label'] != 4]  # Remove label 4

# Create mapping: old_label -> new_label
label_mapping = {}
new_label = 0
for old_label in sorted(dataset['label'].unique()):
    label_mapping[old_label] = new_label
    print(f"  {class_names[old_label]} (old: {old_label}) -> (new: {new_label})")
    new_label += 1

# Reverse mapping: new_label -> old_label (for printing names)
rev_label_mapping = {new: old for old, new in label_mapping.items()}

# Apply mapping
dataset['label'] = dataset['label'].map(label_mapping)

print("\nDataset loaded and converted to DataFrame.")

## 2. Take a uniform 30% of dataset -> min class count is ~150 so unifrom distribution means samples per class are capped to min class count
def select_uniform_subset(df, fraction, label_col="label", seed=SEED, accepted_labels=None):
    
    if accepted_labels is not None:
        # convert to integers
        accepted_labels = [int(x) for x in accepted_labels]
        df = df[df[label_col].isin(accepted_labels)]
    
        if class_names:
            print(f"\n{'='*70}")
            print(f"Filtering to {len(accepted_labels)} selected classes:")
            print(f"{'='*70}")
            for label_id in sorted(accepted_labels):
                # map new label back to original index to get the name
                orig_id = rev_label_mapping.get(label_id)
                class_name = class_names[orig_id] if orig_id is not None else f"label_{label_id}"
                count = len(df[df[label_col] == label_id])
                print(f"  [{label_id:2d}] {class_name:50s} ({count} samples)")
            print(f"{'='*70}")
            print(f"Total samples after filtering: {len(df)}")
            print(f"{'='*70}\n")

    # Creates a dataset ordered by class labels and samples a fraction of the data from each class to form a full subset of the dataset
    subset = df.groupby(label_col, group_keys=False).sample(frac=fraction).reset_index(drop=True)   
    
    # print(f"Selected {len(subset)} rows out of {len(df)} ({per_class} per class with {n_classes} classes).")
    print(f"Selected {len(subset)} rows out of {len(df)}.")
    return subset

subset_uniform = select_uniform_subset(dataset, 1, accepted_labels=['28', '29', '30', '31', '32', '33', '34', '35', '36', '37']) # all claases are equally represented, useful for debugging + initial model dev (?)
subset_random  = dataset.sample(frac=0.3, random_state=SEED).reset_index(drop=True) # Proportional

## 3. Remove empty values
subset_mvp_clean = subset_uniform.dropna(axis=0, how='any')

## 4. Seperate into random train/val/test splits (70/15/15)
mvp_train, mvp_leftover = train_test_split(subset_mvp_clean, train_size=0.7, shuffle=True, random_state=SEED)
mvp_val, mvp_test = train_test_split(mvp_leftover, train_size=0.5, shuffle=True, random_state=SEED)


## 5. Save processed dataset to disk in folder structure for ClearML upload
def save_split_images(df, split_name, base_dir="processed_dataset"):
    for i, row in df.iterrows():
        label = str(row["label"])
        img = row["image"]
        split_dir = os.path.join(base_dir, split_name, label)
        os.makedirs(split_dir, exist_ok=True)
        img_id = f"{i:05d}.png"
        img_path = os.path.join(split_dir, img_id)
        img.save(img_path)
    print(f"Saved {split_name} split with {len(df)} images to '{base_dir}/{split_name}'.")

# Save all splits to disk for ClearML upload
save_split_images(mvp_train, "train")
save_split_images(mvp_val, "val")
save_split_images(mvp_test, "test")

# Upload processed folder to ClearML
dataset = Dataset.create(
    dataset_project="Small Group Project - Team 2",
    dataset_name="Tomatoes-Only",
    dataset_tags=["uniform_sampling", "no_subset"],
    description="Processed PlantVillage dataset of tomatoes only with images organized in train/val/test folders."
)

dataset.add_files(path="processed_dataset")  # add entire directory
dataset.upload()
dataset.finalize()
print("Uploaded processed dataset to ClearML successfully!")