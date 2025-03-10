# Attention Detection 
#### Using openCV, deepface and technologies like convolutional neural networks and pre-trained models (e.g., MobileNet, ResNet). Detects wheter the user is 'Paying attention or not'. 
##### Gathers data from: 
- Facial Orientation
- Eye State (eye tracking)
- HeadPose

## Training Stage: 
### Created a script (branched), which collected data/pictures of user labeled Attentive vs. Distracted. 
#### fire.py -- Collects data:
- uses openCv and deepface to track face
- Saves data by taking a picture every now n then and by added controls (a = attentive, b = distracted)
- To save to different folders --> change the directory in the top couple lines of code
- ENSURE you are being accurate with the data or train_model.py will hate you



![me1](https://github.com/user-attachments/assets/d128a5ac-6119-4525-a65f-1f7c87f61ef9)
![me2](https://github.com/user-attachments/assets/33832f87-4084-4dfc-bcd7-76f2fc51ed21)


#### train_model.py
- uses tensorflow (thanks google) to train the model and save the h5 file. SWAG


## Goal 
### --> to create and train a Model which detects wheter the user is paying attention or not. To be able to deploy this model on different types of projects. 
