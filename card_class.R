################CNN############
#BiocManager::install("EBImage")
library(keras)
library(EBImage)
library(Rfast)


setwd("C:/Protein/CNN Cards/Cards/club")
img.tr<- sample(dir());train<-list(NULL);for(i in 1:length(img.tr))
{ train[[i]]<- readImage(img.tr[i])
  train[[i]]<- resize(train[[i]], 100, 100)}
club<- train

setwd("C:/Protein/CNN Cards/Cards/heart")
img.tr<- sample(dir());train<-list(NULL);for(i in 1:length(img.tr))
  { train[[i]]<- readImage(img.tr[i])
train[[i]]<- resize(train[[i]], 100, 100)}
heart<- train

setwd("C:/Protein/CNN Cards/Cards/spade")
img.tr<- sample(dir());train<-list(NULL);for(i in 1:length(img.tr))
{ train[[i]]<- readImage(img.tr[i])
train[[i]]<- resize(train[[i]], 100, 100)}
spade<- train

setwd("C:/Protein/CNN Cards/Cards/diamond")
img.tr<- sample(dir());train<-list(NULL);for(i in 1:length(img.tr))
{ train[[i]]<- readImage(img.tr[i])
train[[i]]<- resize(train[[i]], 100, 100)}
diamond<- train

train<-c(club[1:40], heart[1:40], spade[1:40], diamond[1:40])
test<-c(club[41:43], heart[41:43], spade[41:43], diamond[41:43])

par(mfrow=c(3,4))
for(i in 1:12){
plot(test[[i]])}

train<-combine(train)
test<-combine(test)

display(test)

train<- aperm(train, c(4,1,2,3))
test<- aperm(test, c(4,1,2,3))
x<-tile(test, 4)
display(x)
#one hot encoding
train_y<-c(rep(0,40),rep(1,40),rep(2,40),rep(3,40))
test_y<-c(rep(0,3),rep(1,3),rep(2,3),rep(3,3))

train_lab<-to_categorical(train_y)
test_lab<-to_categorical(test_y)


#model
model.card.1<- keras_model_sequential()

model.card.1 %>% 
  layer_conv_2d(filters = 40,
                kernel_size = c(4,4),
                activation = 'relu',
                input_shape = c(100,100,4)) %>%
  layer_conv_2d(filters = 40,
                kernel_size = c(4,4),
                activation = 'relu') %>%
  layer_max_pooling_2d(pool_size = c(4,4) )%>%
  layer_dropout(rate = 0.25) %>%
  layer_conv_2d(filters = 80,
                kernel_size = c(4,4),
                activation = 'relu') %>%
  layer_conv_2d(filters = 80,
                kernel_size = c(4,4),
                activation = 'relu') %>%  
  layer_max_pooling_2d(pool_size = c(4,4)) %>%
  layer_dropout(rate = 0.35) %>%
  layer_flatten()%>%
  layer_dense(units = 256, activation = 'relu')%>%
  layer_dropout(rate= 0.25)%>%
  layer_dense(units = 4, activation = "softmax")%>%
  compile(loss = 'categorical_crossentropy',
          optimizer = optimizer_adam(),
          metrics = c("accuracy"))

summary(model.card.1)

#fit model
history<- model.card.1 %>%
  fit(train, 
      train_lab, 
      epochs = 100,
      batch_size = 40,
      validation_split = 0.2
  )
#options(keras.view_metrics = T)
plot(history)

model.card.1%>% evaluate(train, train_lab)
pred<- model.card.1%>% predict_classes(train)
my_tab_train<-table(Predicted = pred, Actual = train_y)
prob<- model.card %>% predict_proba(train)
cbind(prob, predict_class = pred, Actual = trainy)

model.card.1%>% evaluate(test, test_lab)
pred1<- model.card.1 %>% predict_classes(test)
my_tab<-table(Predicted = pred1, Actual = test_y)
prob<- model.card %>% predict_proba(test)
pb<-c(NULL)

rownames(my_tab)<-c("Clubs", "Hearts", "Spades", "Diamonds")
colnames(my_tab)<-c("Clubs", "Hearts", "Spade", "Diamonds")
rownames(my_tab_train)<-c("Clubs", "Hearts", "Spades", "Diamonds")
colnames(my_tab_train)<-c("Clubs", "Hearts", "Spades", "Diamonds")

my_tab
my_tab_train


