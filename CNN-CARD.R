

#install.packages("keras")
library(keras)
#install_keras() #Keras library as well as the TensorFlow backend use the install_keras() function

#if (!requireNamespace("BiocManager", quietly = TRUE))
#install.packages("BiocManager")
#BiocManager::install("EBImage")
library(EBImage)



setwd("C:/Protein/CNN Cards/Cards/club") # To access the images of Clubs suit.
# The path should be modified as per your 
# machine
img.card<- sample(dir()); #-------shuffle the order
cards<-list(NULL);        
for(i in 1:length(img.card))
{ cards[[i]]<- readImage(img.card[i])
cards[[i]]<- resize(cards[[i]], 100, 100)}
club<- cards              # Storing stack of the Clubs cards in
# matrix form in a list
#---------------------------------------------------------
setwd("C:/Protein/CNN Cards/Cards/heart")# To access the images of Hearts suit.
# The path should be modified as per your 
# machine
img.card<- sample(dir());
cards<-list(NULL);
for(i in 1:length(img.card))
{ cards[[i]]<- readImage(img.card[i])
cards[[i]]<- resize(cards[[i]], 100, 100)}
heart<- cards             # Storing stack of the Hearts cards in
# matrix form in a list
#------------------------------------------------------------
setwd("C:/Protein/CNN Cards/Cards/spade")# To access the images of Spades suit.
# The path should be modified as per your 
# machine
img.card<- sample(dir());
cards<-list(NULL);
for(i in 1:length(img.card))
{ cards[[i]]<- readImage(img.card[i])
cards[[i]]<- resize(cards[[i]], 100, 100)}
spade<- cards             # Storing stack of the Spades cards in
# matrix form in a list
#------------------------------------------------------------
setwd("C:/Protein/CNN Cards/Cards/diamond") # To access the images of Diamonds suit.
# The path should be modified as per your 
# machine
img.card<- sample(dir());
cards<-list(NULL);
for(i in 1:length(img.card))
{ cards[[i]]<- readImage(img.card[i])
cards[[i]]<- resize(cards[[i]], 100, 100)}
diamond<- cards           # Storing stack of the Diamonds cards in
# matrix form in a list
#------------------------------------------------------------

train_pool<-c(club[1:40], 
              heart[1:40], 
              spade[1:40], 
              diamond[1:40]) # Vector of all train images. The first
# 40 images from each suit is included
# in train set
train<-aperm(combine(train_pool), c(4,1,2,3)) # Combine and stacked

test_pool<-c(club[41:43], 
             heart[41:43], 
             spade[41:43], 
             diamond[41:43]) # Vector of all train images. The last
# 3 images from each suit is included
# in test set
test<-aperm(combine(test_pool), c(4,1,2,3)) # Combine and stacked

#-----------------------------------------------------------------

#one hot encoding
train_y<-c(rep(0,40),rep(1,40),rep(2,40),rep(3,40))
test_y<-c(rep(0,3),rep(1,3),rep(2,3),rep(3,3))

train_lab<-to_categorical(train_y) #Catagorical vector for training 
#classes
test_lab<-to_categorical(test_y)#Catagorical vector for test classes

#-------------------------------------------------------------------



# Model Building
model.card<- keras_model_sequential() #-Keras Model composed of a 
#-----linear stack of layers
model.card %>%                  #---------Initiate and connect to #------------------------------------------------------------------#
  layer_conv_2d(filters = 40,       #----------First convoluted layer
                kernel_size = c(4,4),             #---40 Filters with dimension 4x4
                activation = 'relu',              #-with a ReLu activation function
                input_shape = c(100,100,4)) %>%   
  #-----------------------------------------------------------------#
  layer_conv_2d(filters = 40,       #---------Second convoluted layer
                kernel_size = c(4,4),             #---40 Filters with dimension 4x4
                activation = 'relu') %>%          #-with a ReLu activation function
  #-----------------------------------------------------------------#
  layer_max_pooling_2d(pool_size = c(4,4) )%>%   #--------Max Pooling
  #-----------------------------------------------------------------#
  layer_dropout(rate = 0.25) %>%   #-------------------Drop out layer
  #-----------------------------------------------------------------#
  layer_conv_2d(filters = 80,      #-----------Third convoluted layer
                kernel_size = c(4,4),            #----80 Filters with dimension 4x4
                activation = 'relu') %>%         #--with a ReLu activation function
  #-----------------------------------------------------------------#
  layer_conv_2d(filters = 80,      #-----------Third convoluted layer
                kernel_size = c(4,4),            #----80 Filters with dimension 4x4
                activation = 'relu') %>%         #--with a ReLu activation function
  #-----------------------------------------------------------------#
  layer_max_pooling_2d(pool_size = c(4,4)) %>%  #---------Max Pooling
  #-----------------------------------------------------------------#
  layer_dropout(rate = 0.35) %>%   #-------------------Drop out layer
  #-----------------------------------------------------------------#
  layer_flatten()%>%   #---Flattening the final stack of feature maps
  #-----------------------------------------------------------------#
  layer_dense(units = 256, activation = 'relu')%>% #-----Hidden layer
  #-----------------------------------------------------------------#
  layer_dropout(rate= 0.25)%>%     #-------------------Drop-out layer
  #-----------------------------------------------------------------#
  layer_dense(units = 4, activation = "softmax")%>% #-----Final Layer
  #-----------------------------------------------------------------#
  
  compile(loss = 'categorical_crossentropy',
          optimizer = optimizer_adam(),
          metrics = c("accuracy"))   # Compiling the architecture

#---------------------------------fit model----------------------------
history<- model.card %>%
  fit(train, 
      train_lab, 
      epochs = 100,
      batch_size = 40,
      validation_split = 0.2
  )
#------------------------------------------------------------------------

model.card %>% evaluate(train,train_lab) #Evaluation of training set pred<- model.card %>% predict_classes(train) #-----Classification
pred<- model.card%>% predict_classes(train)
Train_Result<-table(Predicted = pred, Actual = train_y) #----Results

#prob<- model.card %>% predict_proba(train)

model.card %>% evaluate(test, test_lab) #-----Evaluation of test set
pred1<- model.card  %>% predict_classes(test)   #-----Classification
Test_Result<-table(Predicted = pred1, Actual = test_y) #-----Results
rownames(Train_Result)<-rownames(Test_Result)<-colnames(Train_Result)<-colnames(Test_Result)<-c("Clubs", "Hearts", "Spades", "Diamonds")

print(Train_Result)
print(Test_Result)




