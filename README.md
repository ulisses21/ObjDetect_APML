# ObjDetect_APML
# # Run Inference: 

Run the file model "model Yolo_pred_vid" or "model Yolo_pred_webcam"

this will use the "person_w.pt" model that was trained for 3 epochs with 1743 images 

(this model is a "test run only" confidence in detecting human agaisnt human is 0.23 when looking a the confusion matrix)

to see the result open the folder runs/detect/predict


# # Train model:

Run the "data.py" this will download the dataset to your PC 

(if you wish to modify the number of images downloaded change the fraction variable)(currently at 0.001 (1743 images))

Run the "model Yolo_train.py" to train the model

this will create a folder path called runs/detect with a lot of train subfolders, inside the weights folder.

look for the "best.pt" file this is your trained model  

copy it to the main folder and name it what you wish(name.pt)

after that replace the  'person_w.pt' with your 'name.pt' model in the "model Yolo_pred_vid" or "model Yolo_pred_webcam" files
and Run the file.


