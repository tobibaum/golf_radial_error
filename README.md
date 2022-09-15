# Example Evaluation of putting study in golf

Goal: measure the distance of the golf put to the hole. white card = ball hit hole. blue card = ball missed entirely

Dimensions and conversion of pixel to meters in `golf_utils.py`

WARNING: method has been developed for full-size images. Here we use 640x480. for best results, implement better Vision and finetuning the algorithms ;)

### Run
`0_golf_simple.ipynb`: simple color-based location of golf ball

`1_evaluate_pics.ipynb`: full evaluation of sample data


## Example
The input images are taken with a GoPro and must therefore be distorted.!

Input: raw images
![Screenshot from 2022-09-15 18-09-15](https://user-images.githubusercontent.com/1063330/190453826-4f12ede8-52b6-4808-b68a-6f46ae9de0f2.png)


Output: straight images with hit/miss or distance annotation
![P05_QE_Post_j=127](https://user-images.githubusercontent.com/1063330/190453589-0f5588d2-51c4-4675-8509-283765841f59.png)
