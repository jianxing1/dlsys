from typing import List, Optional
from ..data_basic import Dataset
import numpy as np
import gzip
import struct

class MNISTDataset(Dataset):
    def __init__(
        self,
        image_filename: str,
        label_filename: str,
        transforms: Optional[List] = None,
    ):
        ### BEGIN YOUR SOLUTION
        super().__init__(transforms)
        self.image_filename = image_filename
        self.label_filename = label_filename
        self.X, self.y = parse_mnist(self.image_filename, self.label_filename)
        self.X = self.X.reshape(self.X.shape[0], 28, 28, 1)
        ### END YOUR SOLUTION

    def __getitem__(self, index) -> object:
        ### BEGIN YOUR SOLUTION
        return (self.apply_transforms(self.X[index]), self.y[index])
        ### END YOUR SOLUTION

    def __len__(self) -> int:
        ### BEGIN YOUR SOLUTION
        return self.X.shape[0]
        ### END YOUR SOLUTION

def parse_mnist(image_filename, label_filename):
    """ Read an images and labels file in MNIST format.  See this page:
    http://yann.lecun.com/exdb/mnist/ for a description of the file format.

    Args:
        image_filename (str): name of gzipped images file in MNIST format
        label_filename (str): name of gzipped labels file in MNIST format

    Returns:
        Tuple (X,y):
            X (numpy.ndarray[np.float32]): 2D numpy array containing the loaded 
                data.  The dimensionality of the data should be 
                (num_examples x input_dim) where 'input_dim' is the full 
                dimension of the data, e.g., since MNIST images are 28x28, it 
                will be 784.  Values should be of type np.float32, and the data 
                should be normalized to have a minimum value of 0.0 and a 
                maximum value of 1.0 (i.e., scale original values of 0 to 0.0 
                and 255 to 1.0).

            y (numpy.ndarray[dtype=np.uint8]): 1D numpy array containing the
                labels of the examples.  Values should be of type np.uint8 and
                for MNIST will contain the values 0-9.
    """
    ### BEGIN YOUR CODE
    with gzip.open(image_filename, 'rb') as img_file:
      magic_number, image_num, rows, cols = struct.unpack('>4i', img_file.read(16))
      assert(magic_number == 2051)
      total_pixels = rows * cols
      X = np.vstack([np.array(struct.unpack(f"{total_pixels}B", img_file.read(total_pixels)), 
          dtype = np.float32) for _ in range(image_num)])
      X_max = np.max(X)
      X_min = np.min(X)
      X = (X - X_min) / (X_max - X_min)
      # print(X)

    with gzip.open(label_filename, 'rb') as label_file:
      magic_number, num = struct.unpack('>2i', label_file.read(8)) 
      assert(magic_number == 2049)
      y = np.array(struct.unpack(f"{num}B", label_file.read(num)), dtype = np.uint8)
      # print(y)
    return(X, y)
    ### END YOUR CODE

    