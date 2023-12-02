img_dataset.py can be seen as the script equivalent of the notebook
i go on more detail in the notebook how i experimented and came up with a satisfactory dataloader

* Please feel free to read the exported notebook.pdf file(only, as its enough) 

The notebook goes over making of:
    * RedWebDataset
    * Transforms:
        * Rescale
        * Random Crop
        * To Tensor
        * Composition of transformation
    * Dataloader with batch and shuffling support

Boils down to making of :
```py
dataloader = DataLoader(transformed_dataset, batch_size=4, shuffle=True, num_workers=4)
```
