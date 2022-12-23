# ExoGlove Object Detection Implementation in Pytorch

This repo implements SSD (Single Shot MultiBox Detector) in PyTorch for object detection, using MobileNet backbones. It also has out-of-box support for retraining on a private ExoGlove dataset and a fruit dataset.  
> Thanks to @qfgaohao for the upstream implementation from:  [https://github.com/qfgaohao/pytorch-ssd](https://github.com/qfgaohao/pytorch-ssd)

## Dependencies
1. Python 3.9+
2. OpenCV
3. Pytorch 1.13+
4. Pandas
5. Argparse
### Dependencies Installation
1. Install [Anaconda](https://www.anaconda.com/)
2. Install environment
    ```bash
    conda env create -f environment.yml
    ```
3. Activate environment
    ```bash
    conda activate ExoGlove
    ```

## Run the demo


### Run the live demo
#### CLI Parameters
| Parameters            | Description | Value Choices |
| ----------            | ----------- | ------------- |
| --input-source-path   | Absolute Path/Relative Path of the test video | 

#### Commend
```bash
python run_ssd_example_video.py --input-source-path [PATH_OF_VIDEO]
```


### Test on image

#### CLI Parameters
| Parameters            | Description | Value Choices |
| ----------            | ----------- | ------------- |
| --input-source-path   | Absolute Path/Relative Path of the test image | 

#### Commend
```bash
python run_ssd_example_image.py --input-source-path [PATH_OF_IMAGE]
```


### Train
#### CLI Parameters
The ExoGlove-Fruit dataset and the model file can be download from https://sfsu.box.com/s/z2p9usaxxhphpt8s6elkde66oeptqlh6  
The dataset need to be place on data folder  
The model file need to be place on models folder  
| Parameters            | Description | Value Choices |
| ----------            | ----------- | ------------- |
| --net   | Network Type(default is mb2-ssd-lite) | mb2-ssd-lite
| ----pretrained-ssd   | Absolute/Relative Path of the pretrained model weight(default is models\mb2-ssd-lite-mp-0_686-pretrained-VOC.pth)  | 
| --lr   | initial learning rate(default is 0.01)  | 
| --momentum   | Momentum value for optim(default is 0.9)  | 
| --weight-decay   | Weight decay for SGD(default is 5e-4)  | 
| --gamma   | Gamma update for SGD(default is 0.1)  | 
| --scheduler   | Scheduler for SGD. It can one of multi-step and cosine(default is cosine)  | multi-step,cosine
| --t-max   | T_max value for Cosine Annealing Scheduler(default is 100)  | 
| --batch-size   | Batch size for training(default is 4)  | 
| --num-epochs   | the number epochs(default is 200)  | 
| --num-workers   | Number of workers used in dataloading(default is 2)  | 
| --use-cuda   | Use CUDA to train model(default is true)  | 
| ----model-dir   | Directory for saving checkpoint models(default is models/my_model)  | 

#### Commend
```bash
python train_ssd.py
```
#### Caution
**If you are a windows/Mac User, please set the num-workers to 0.**

## Convert to ONNX Model
### CLI Parameters
| Parameters            | Description | Value Choices |
| ----------            | ----------- | ------------- |
| --net   | Network Type(default is mb2-ssd-lite) | mb2-ssd-lite
| --input   | Absolute/Relative Path of the model weight(default is models\model_exoglove_07262022.pth)  | 
| --output   | Absolute/Relative Path of the output onnx file(default is models\model_exoglove_07262022.onnx)  | 
| --labels   | Absolute/Relative Path of the labels file(default is models\labels.txt)  | 

### commend
```bash
python export_onnx.py
```

## Deploy on Jetson Nano
> Refer to [Locating Objects with DetectNet](https://github.com/dusty-nv/jetson-inference/blob/master/docs/detectnet-console-2.md)