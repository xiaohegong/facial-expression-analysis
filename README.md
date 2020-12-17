# Facial Expression Recognition

### How to train model
* Run `python save_data.py` to store data in `./data/CK_data.h5` (Different model require you to run different save_* file)
* Using the the file with prefix train_*, for example, `python train_vgg16.py` to train the model, you can store the model by change dst_path variable.

### How to run camera app:

* Package Requirement: tkinter, cv2, PIL, numpy, pytorch, argparse, imutils, dlib

* flag: `--model ./path/to/your/model`

* Example: `python camera.py --model ./model_data/vgg16.pt`

* Example interface:

  <img src="./result/camera_app_example.PNG" style="zoom:50%;" />

* Example snapshot(take photo button):
  
  <img src="./result/IMG-20-12-13-14-12.jpg" style="zoom:50%;" />

  <img src="./result/IMG-16-20-48-17-12.jpg" style="zoom:50%;" />

  <img src="./result/IMG-16-21-39-17-12.jpg" style="zoom:50%;" />
