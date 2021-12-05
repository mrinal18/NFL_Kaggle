# Introduction
The National Football League is America’s most popular sports league. Founded in 1920, the NFLdeveloped the model for the successful modern sports league and is committed to advancing progressin the diagnosis, prevention, and treatment of sports-related injuries.  Health and safety efforts in-clude support for independent medical research and engineering advancements as well as a com-mitment  to  work  to  better  protect  players  and  make  the  game  safer,  including  enhancements  tomedical protocols and improvements to how our game is taught and played. This work was done foranalysing sports injury and mitigating this problem by surveying them.  there are few novel worksbefore where they detected the helmet impacts, but here they are taking it to the next level by as-signing each player to each helmet and accurately identity player’s exposure throughout footballplaysCurrently, NFL annotates each subset of players each year to check the exposure and to expand thisfeature they require the field view to determine each and every players positions.  This competition will be a subset of the previous year’s competition from kaggle.com, and we will be using almostsame dataset with more information given to use on dataset.  Along with information, each playeris associated with videos, showing a sideline and endzone views which are aligned so that frames corresponds between the two videos. To aid with helmet detection, we are also provided an ancillarydataset of images showing helmets with labeled bounding boxes.   This year they also providing baseline helmet detection boxes for the training and test set. As per the rule after submission of thiscode, they will run the submitted model in 15 unseen plays that is a part of a holdout traning set.During the competition participants are evaluated on the test set and a part of the predictions is usedto calculate the above mentioned metrics and show the best score for each participant on a publicleaderboard. After the competition the score with respect to the remaining part is released and usedto determine and display the final scoring (private leaderboard).  For training our models we usedPytorch with object detection algorithms with tracking. Moreover, we use several implementations.

# Implementation 

The Notebooks are divided in following manner in our page:
 
 * EDA
 * MODEL
 * Evaluation


## EDA:
The  main  task  before  we  even  start  our  modelling  part  is  to  first  understand  our  data  and  inferwhat we can extract from that information.   First we looked into the basic information on whatdata is being given to us and how we will represent those data.  we came across information likethe bounding box information has been provided for each player.  Our main training data consistof playerid,  x-y coordinate of those players,  direction of the players,the snap information,  teaminformation and which frame they belong to.  While trying to understand the training and testinginformation, we found that testing data is a subset of training information.  We have 120 trainingvideos out of which 60 are sidezone and 60 are endzone images (8).  Does Sideline and Endzone Video has same frame?  No, 25 plays out of 60 doesn’t match and the difference is mostly 1 framebut there are 7 frame difference also. We checked even the biggest and smallest size of the helmet,that means that with respect to the frame,  how much percentage is being taken by an individual helmets.

## Model Implementation: 
Since we are working on the multi object detection,  we started using a model that works really powerfully while detecting the images, hence, we used YOlOv5 for our case, but along with this wealso wanted to use algorithms for tracking the players in our case.  This gave rise to a combination of YOLOv5 and DeepSort algorithms.  In this section we will be discussing about various methodsthat we used to make our final model work. 
Our solution consist of the following stages 

  *  Use YOLOv5 for detecting helmets.
  *  Use deepsort to track the helmet.
  *  Tracking Players:
      *  Use ICP (Iterative Closest Point) and the Hungarian algorithm to assign helmet boxesto the tracking data for each frame.
      *  Divide helmets into two clusters using k-means based on the color of the helmet.
      *  Use ICP (Iterative Closest Point) and the Hungarian algorithm to assign helmet boxesto the tracking data again, taking into account the results of deepsort and the helmetcolor information.
  * 4.  Use the Hungarian algorithm to determine the final label


## Evaluation:
Please refer to Evaluation notebooks in Eval directory, we implemented all our metrics using Weights and Bais Dashboards 



