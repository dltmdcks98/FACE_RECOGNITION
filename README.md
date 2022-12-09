개발 환경
VSCode, Visual Studio 2022(C++), CMAKE, python(3.99), Git

성과 
openCV와 python을 다루고, 기본적인 이미지 구별에 대한 지식을 얻었다. 

개선점  
1. 데이터를 축적하고 학습하는 법을 학습하여 보다 높은 정확도를 갖도록 한다.
=> https://github.com/ageitgey/face_recognition/blob/master/examples/face_recognition_knn.py 
2. 1번과 유사한데 비슷한 사람을 구분하기 까지의 데이터를 축적하여 저장할 서버를 구축해야한다.


Reference:
1. https://hslee09.medium.com/python-cnn%EC%9D%84-%ED%99%9C%EC%9A%A9%ED%95%9C-%EC%97%B0%EC%98%88%EC%9D%B8-%EC%82%AC%EC%A7%84-%EB%B6%84%EB%A5%98-1-705aee34def4
2. https://ukayzm.github.io/unknown-face-classifier/#%EC%96%BC%EA%B5%B4-%EC%9D%B8%EC%8B%9D
3. https://wiserloner.tistory.com/1123
4. https://www.youtube.com/watch?v=sz25xxF_AVE



CNN vs KNN

CNN : 영상처리 딥러닝에서 사용
KNN : 분류를 위한 알고리즘 
https://www.inflearn.com/questions/89475

REF: https://wonwooddo.tistory.com/47



---------------------------------------------------------------------------------------------
file : face_recognition_knn.py , source => https://github.com/ageitgey/face_recognition/blob/master/examples/face_recognition_knn.py

This is an example of using the k-nearest-neighbors (KNN) algorithm for face recognition.

When should I use this example?
This example is useful when you wish to recognize a large set of known people,
and make a prediction for an unknown person in a feasible computation time.

Algorithm Description:
The knn classifier is first trained on a set of labeled (known) faces and can then predict the person
in an unknown image by finding the k most similar faces (images with closet face-features under euclidean distance)
in its training set, and performing a majority vote (possibly weighted) on their label.

For example, if k=3, and the three closest face images to the given image in the training set are one image of Biden
and two images of Obama, The result would be 'Obama'.

* This implementation uses a weighted vote, such that the votes of closer-neighbors are weighted more heavily.

Usage:

1. Prepare a set of images of the known people you want to recognize. Organize the images in a single directory
   with a sub-directory for each known person.

2. Then, call the 'train' function with the appropriate parameters. Make sure to pass in the 'model_save_path' if you
   want to save the model to disk so you can re-use the model without having to re-train it.

3. Call 'predict' and pass in your trained model to recognize the people in an unknown image.

NOTE: This example requires scikit-learn to be installed! You can install it with pip:

$ pip3 install scikit-learn
