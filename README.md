# JumpTracker
Uses tensor flow js to detect faces and trackes them with the similarities put into munkres. When a face moves up enough it will print out jump. 
The are faces are tracked as follows:
For each frame:
1. Tensor flow model detects face
2. Each detection gets a similarity to the previous frame detections using matrix interestion over union
3. The similarities are put into the munkres algrorithm which matches them up to the most similar detection

## Development 
```
yarn start
```
to run the server and the web app. 
It take a little while for the model to be pulled into the application before face detection.

## To Do
* typescript
* fix bug of double jump
* make it an extension to play the dino jump game
