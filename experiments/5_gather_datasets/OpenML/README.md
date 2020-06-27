# Usage

```
python select_datasets.py
```
This selects OpenML datasets, output indices into a csv file and selection criteria into a txt file.

To print out OpenML dataset indices together with dataset name and size information, do
```
python select_datasets.py details
```

```
./gather_datasets.sh
```
This gathers OpenML datasets whose indices come from `selected_dataset_indices.csv`. Output directory should be specified in `generate_preprocessed_dataset.py`.