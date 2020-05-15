rm(list = ls()) # clear global environment
graphics.off() # close all graphics
library(pacman) # needs to be installed first
# p_load is equivalent to combining both install.packages() and library()
p_load(knitr, dplyr, kableExtra, sampler, magrittr, caret, rpart, 
caretEnsemble,parallel,doParallel, recipes,  e1071)

setwd("~/finalProject/") # Set directory on OSC server

# Read in datasets
# The original paper specifically looks at week 1 internal data
# and week 3 external traffic data.
external3 <- read.csv("CIDDS-001-external-week3.csv") 
internal1 <- read.csv("CIDDS-001-internal-week1.csv") 

# Randomly select 172839 instances from week 1 traffic data
i1.sample <- rsamp(internal1, 172839)

# Drop unused columns and change data type of 'Bytes' from Factor to numeric
# "We have neglected AttackID and AttackDescription features because they just 
# give more information about executed attacks. Hence these attributes do not
# play any important role in classification"
# I also remove the Flows and Tos features because they were not included in
# the original paper.
external3 %<>% select(-Flows, -Tos, -attackID, -attackDescription) %>% 
    mutate(Bytes = as.numeric(Bytes))
i1.sample %<>% select(-Flows, -Tos, -attackID, -attackDescription) %>% 
    mutate(Bytes = as.numeric(Bytes))
                      
# Partition data into 66% training and 34% testing
e3.trainRowNums = createDataPartition(external3$class, p=0.66, list = F)
e3.trainData = external3[e3.trainRowNums,]
e3.testData = external3[-e3.trainRowNums,]

i1.trainRowNums = createDataPartition(i1.sample$class, p=0.66, list = F)
i1.trainData = i1.sample[i1.trainRowNums,]
i1.testData = i1.sample[-i1.trainRowNums,]

# Ensure the training and testing data have similar proportions in the 'class'
# attribute as the original data.

# 'class' proportions in external week 3 training data
prop.table(table(e3.trainData$class)) 
# 'class' proportions in external week 3 testing data
prop.table(table(e3.testData$class))
# 'class' proportions in external week 3 original data
prop.table(table(external3$class))

# 'class' proportions in internal week 1 training data
prop.table(table(i1.trainData$class))
# 'class' proportions in internal week 1 testing data
prop.table(table(i1.testData$class))
# 'class' proportions in internal week 1 sampled data
prop.table(table(i1.sample$class))
# 'class' proportions in internal week 1 original data
prop.table(table(internal1$class))

# Preprocess data
e3.model_recipe <- recipe(class ~ ., data = e3.trainData)
e3.model_recipe_steps <- e3.model_recipe %>% 
    step_other(
    threshold = 0.1,
    all_nominal(),
    -all_outcomes())
prepped_recipe <- prep(e3.model_recipe_steps, training = e3.trainData)
e3.trainData.preProcessed <- bake(prepped_recipe, e3.trainData)
e3.testData.preProcessed <- bake(prepped_recipe, e3.testData)

i1.model_recipe <- recipe(class ~ ., data = i1.trainData)
i1.model_recipe_steps <- i1.model_recipe %>% 
    step_other(
    threshold = 0.1,
    all_nominal(),
    -all_outcomes())

# Separate the 'class' column from the testing data for both datasets
y.e3.test = e3.testData.preProcessed$class
y.i1.test = i1.testData$class

# Model building using 66% partitioned testing data
numCores = 16 # Using 16 cores
cl = makePSOCKcluster(numCores, outfile="niddModelLog.txt")
registerDoParallel(cl)

e3.trControl = trainControl(
  method = "cv",
  number = 10,
  classProbs = T,
  selectionFunction = "best",
  index = createResample(e3.trainData.preProcessed$class, 10) )

i1.trControl = trainControl(
  method = "cv",
  number = 10,
  classProbs = T,
  selectionFunction = "best",
  index = createResample(i1.trainData$class, 10) )

#"Statistical analysis of CIDDS-001 can be done using other ML algorithms in 
# order to analyse their performance on the dataset."  
# CART classification model for week 3 external server traffic data
e3.cartFit <- train(class ~ ., data=e3.trainData.preProcessed, method="rpart", 
    trControl = e3.trControl, tuneLength = 10)
# CART classification model for week 1 openstack traffic data
i1.cart <- train(i1.model_recipe_steps, data = i1.trainData, method="rpart",
    trControl = i1.trControl, tuneLength=10)

# Classify external week 3 traffic data using CART
e3.cartPredict <- predict(e3.cartFit, newdata = e3.testData.preProcessed)
confusionMatrix(e3.cartPredict, y.e3.test)
# We can be confident that the accuracy for the external week 3 traffic
# CART model is between 97.09% and 97.37%. It seems as though the 
# 'attacker' and 'victim' classes are identified with 100% accuracy.

# Classify week 1 openstack traffic data using CART
i1.cartPredict <- predict(i1.cart, newdata = i1.testData, type = "raw")
confusionMatrix(i1.cartPredict, y.i1.test)
# The CART model for the OpenStack traffic data accuracy is 100%, which as
# mentioned in the original paper "may be due to random sampling of instances
# from the dataset file which can lead to some biased instance selections."

stopCluster(cl)