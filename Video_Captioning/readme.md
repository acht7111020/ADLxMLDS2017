# Video Caption
* Project link: [link](https://www.csie.ntu.edu.tw/~yvchen/f106-adl/A2)
* Requirement: Given the video features, predict one corresponding caption

## Dataset
* [MSVD dataset](https://drive.google.com/file/d/0B18IKlS3niGFNlBoaHJTY3NXUkE/view)
* Each video has several captions
* The features of each video is 80\*4096 (Using VGG19)

## Pre-processing
1. Create a dictionary to save the id correnponds to features, and create a dictionary to save the id corresponds to captions (prevent out of memory)
2. Every training step uses random sample method to choose training data

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

