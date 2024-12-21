# DSCNet+ (IEEE Transtions on Instrument and Measurement 2023)
[Novel Dilated Separable Convolution Networks for Efficient Video Salient Object Detection in the Wild](https://ieeexplore.ieee.org/abstract/document/10210391)

## 2. Requirements

 - Python 3.7, Pytorch 1.7, Cuda 10.1
 - Test on Win10 and Ubuntu 16.04

## 3. Data Preparation

 - Upload the dataset and trained model (epoch_100.pth). Then put the dataset under the following directory:
 
        -dataset\ 
          -DAVIS\  
          -FBMS\
          ...
        -pretrain
          -epoch_100.pth
          ...

	  
## 4. Testing

    Directly run test.py
    
    The test maps will be saved to './resutls/'.

- Evaluate the result maps:
    
    You can evaluate the result maps using the tool in [Matlab Version](http://dpfan.net/d3netbenchmark/) or [Python_GPU Version](https://github.com/zyjwuyan/SOD_Evaluation_Metrics).
    
 
## 6. Citation

Please cite the following paper if you use this repository in your reseach

@article{
    singh2023novel,
  title={Novel dilated separable convolution networks for efficient video salient object detection in the wild},
  author={Singh, Hemraj and Verma, Mridula and Cheruku, Ramalingaswamy},
  journal={IEEE Transactions on Instrumentation and Measurement},
  year={2023},
  publisher={IEEE}
}
