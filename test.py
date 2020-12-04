import numpy as np
import torch as T

from model.cnn import CNN

if __name__ == "__main__":
    model_path = "model_data/cnn.pt"
    model = T.load(model_path)
    model._test()



