# Video Caption
* Project link: [link](https://www.csie.ntu.edu.tw/~yvchen/f106-adl/A2)
* Requirement: Given the video features, predict one corresponding caption

## Dataset
* [MSVD dataset](https://drive.google.com/file/d/0B18IKlS3niGFNlBoaHJTY3NXUkE/view)
* Each video has several captions
* The features of each video is 80\*4096 (Using VGG19)

## Pre-processing
* For training: Create a dictionary to save the id corresponds to captions(train_caption_dict.pkl) (prevent out of memory)
* For testing: Create a dictionary to save the id correnponds to features(test_id.pkl)
* Every training step uses random sample method to choose training data

## Train
1. Download MSVD dataset (NOTICE: remember the path of the video features folders **'\*/feat'**)
2. Follow pre-processing method above to create correnponding files
3. For training, modify ```main.py``` line 265 to change the path of video features folder, and modify line 266 to change the path of pre-processing file(train_caption_dict.pkl)
4. Run ```python3 train.py --train --input [MSVD_folder_path] --savepath [save_model_path]```


## Result
* I get the rank 30th/189 in this course

#### Test
1.
<img src="result/6.gif" height="300px">
Generated caption: three people are dancing <br/>
2.
<img src="result/4.gif" height="300px">
Generated caption: a man is playing a guitar <br/>
3.
<img src="result/8.gif" height="300px">
Generated caption: a cat is looking at a mirror 

