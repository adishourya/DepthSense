## Modelling and Triaining Loop
* The modelling was done as per the suggestions from the ReDWeb_V1 paper
* we utilize a pre-trained resnet50 and only train the last layer to adjust our output shape
* we also employ feature fusion starting from third layer.
* the training log is inside log.txt

# recover the model
Note Github does not allow >=100 mb file to upload. since the model paramaters saved file is ~ 195 mb
They are split into ./redweb_partaa and ./redweb_partab
```sh
# merge the file to make the model by running:
cat redweb_part* > redweb.pt
```
