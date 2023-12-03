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

## the OnlineSampling.ipynb goes over making of online mini batch sampling data loader
* Read about an example and math of how mini batch sampling works in ../../online_sampling/OnlineSampling.pdf
* Read about how i made this dataloader in ./OnlineLoader.pdf

Boils down to
```py
online_dataset = OnlineRedWeb(root_dir="../ReDWeb_V1",transform=transforms.Compose([
    Rescale(256),
    RandomCrop(225),
    ToTensor()
]))
online_loader = DataLoader(online_dataset,batch_size=32,shuffle=True,collate_fn=custom_collate)
```
