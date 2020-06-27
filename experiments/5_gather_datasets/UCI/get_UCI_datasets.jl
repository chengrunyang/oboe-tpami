using UCIData
classification_datasets = UCIData.list_datasets("classification");
for dataset_name in classification_datasets
    UCIData.dataset(dataset_name)
end
