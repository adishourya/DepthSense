## The following script is taken and modified from https://github.com/cs231n/cs231n.github.io

# This mounts your Google Drive to the Colab VM.
```py
from google.colab import drive
drive.mount('/content/drive')
```

# TODO: Enter the foldername in your Drive where you have saved the unzipped
# assignment folder, e.g. 'cs231n/assignments/assignment1/'
```py
FOLDERNAME = DepthSense
assert FOLDERNAME is not None, "[!] Enter the foldername."
```

# Now that we've mounted your Drive, this ensures that
# the Python interpreter of the Colab VM can load
# python files from within it.
import sys
sys.path.append('/content/drive/My Drive/{}'.format(FOLDERNAME))

# This downloads the CIFAR-10 dataset to your Drive
# if it doesn't already exist.
%cd /content/drive/My\ Drive/$FOLDERNAME/cs231n/datasets/
!bash get_datasets.sh
%cd /content/drive/My\ Drive/$FOLDERNAME
