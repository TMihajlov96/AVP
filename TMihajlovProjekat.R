
#Instalacija i ucitavanje bibloteka
#install.packages("ggplot2")
library(ggplot2)
#install.packages("ggeasy")
library(ggeasy)
#install.packages("caret")
library(caret)
#install.packages("dplyr")
library(dplyr)
#install.packages("data.table")
library(data.table)
#install.packages("quanteda")
library(quanteda)
#install.packages("wordcloud")
library(wordcloud)
#install.packages("tidytext")
library(tidytext)
#install.packages("tokenizers")
library(tokenizers)
#install.packages("tm")
library(tm)
#install.packages("tidyr")
library(tidyr)
#install.packages(igraph)
library(igraph)
#install.packages(glue)
library(glue)
#install.packages("psych")
library(psych)
#install.packages("plotly")
library(plotly)
#install.packages("e1071")
library(e1071)
#install.packages("glmnet")
library(glmnet)
#install.packages("quanteda.textmodels")
library(quanteda.textmodels)

#Dodatna podesavanja
options(scipen = 999)

#Ucitavanje podataka
data.og <- read.csv("data/covid.csv", stringsAsFactors = FALSE)
View(data.og)

#Izdvajanje baze za dalji rad
data <- data.og[1:5000, ]
data <- subset(data, select = c (UserName,
                                 Location,
                                 TweetAt,
                                 OriginalTweet,
                                 Sentiment))
data <- as.data.frame(data)

#Kreiranje sentimenta sa 3 kategorije
data$Sentiment <- replace(data$Sentiment, 
                          which(data$Sentiment 
                                == "Extremely Negative"), 
                          "Negative")
data$Sentiment <- replace(data$Sentiment, 
                          which(data$Sentiment 
                                == "Extremely Positive"), 
                          "Positive")
unique(data$Sentiment)

#Kreiranje numericke verijable za 3 kategorije sentimenta
data$SentimentNum <- data$Sentiment
data$SentimentNum <- replace(data$SentimentNum, 
                             which(data$SentimentNum 
                                   == "Negative"), 
                             -1)
data$SentimentNum <- replace(data$SentimentNum, 
                             which(data$SentimentNum
                                   == "Positive"), 
                             1)
data$SentimentNum <- replace(data$SentimentNum, 
                             which(data$SentimentNum 
                                   == "Neutral"), 
                             0)

data$SentimentNum <- as.numeric(data$SentimentNum)

#Kreiranje varijable sa duzinom tvitova
data$TweetLen <- count_characters(data$OriginalTweet)
head(data)

#Deskriptivna statistika baze podataka
describe(data, na.rm = TRUE, interp=FALSE,skew = FALSE, ranges = TRUE,trim=.1,
         type=3,check=TRUE,fast=NULL,quant=TRUE, 
         IQR=FALSE,omit=TRUE,data=NULL) 

#Duzina tvita u odnosu na sentiment 
shapiro.test(data$TweetLen)
kruskal.test(data$TweetLen, data$Sentiment)

data %>%
  filter(!is.na(TweetLen)) %>%
  ggplot(aes(x= Sentiment, y= TweetLen)) +
  geom_boxplot() + 
  ggtitle("Duzina tvita u odnosu na sentiment") + 
  xlab("Sentiment") +
  ylab("Duzina tvita")

#Skor senetimenata u korpusu 
table(data$Sentiment)

data %>% 
  ggplot(aes(x=Sentiment)) +
  geom_bar(aes(fill = Sentiment)) +
  ggtitle("Prikaz tipova sentimenta u 5000 tvitova") +
  xlab("Sentiment") +
  ylab("Broj tvitova") +
  scale_fill_brewer(palette="Set3")

#Broj tvitova po danu
data$TweetAt <- as.Date(data$TweetAt, format = "%d/%m/%y")
dates <- as.data.frame(data$TweetAt) 
dates %>%
  ggplot(aes(x=data$TweetAt)) + 
  stat_count(geom="line", aes(y=..count..)) +
  theme_light() +
  xlab(label = "Datum") +
  ylab(label = NULL) +
  ggtitle(label = "Broj tvitova po danu")

#Pregled lokacija sa kojih je tvitovano 
locations <- removeWords(data$Location, stopwords("SMART"))

wordcloud(locations, min.freq = 1, max.words = 100, scale = c(2.2,1),
          stopwords = TRUE ,colors=brewer.pal(8, "Set3"), random.color = T, 
          random.order = F)

#Podela na test i trening set 
train.indices <- createDataPartition(data$Sentiment, p=0.8, list = FALSE)
traindata <- data[train.indices,]
testdata <- data[-train.indices,]

#Kreiranje trening korpusa
corpus.train <- corpus(traindata$OriginalTweet)

#Kreiranje test korpusa
corpus.test <- corpus(testdata$OriginalTweet)

set.seed(123)

wordcloud(corpus.train, min.freq = 1, max.words = 100, scale = c(2.2,1),
          stopwords = TRUE ,colors=brewer.pal(8, "Accent"), random.color = T, 
          random.order = F)

#Kreiranje liste stop reci
stopwords.tweet <- c("covid2019", "covid_19", "covid19",
                     "coronaviruspandemic", "covid-19", "covid",
                     "corona", "coronavirus", "amp", "t.co", "https")

#Tokenizacija trening seta
tokens.train <- tokens(corpus.train,
                      what = "word1",
                      remove_punct = TRUE,
                      remove_symbols = TRUE,
                      remove_numbers = TRUE,
                      remove_url = TRUE)

#Dodatno sredivanje tokena
tokens.train <- tokens.train %>%
  tokens_tolower() %>%
  tokens_remove(stopwords("SMART")) %>%
  tokens_remove(stopwords.tweet) %>%
  tokens_keep(min_nchar = 2) %>%
  tokens_wordstem(language = "english")

#Tokenizacija test seta
tokens.test <- tokens(testdata$OriginalTweet,
                       what = "word1",
                       remove_punct = TRUE,
                       remove_symbols = TRUE,
                       remove_numbers = TRUE,
                       remove_url = TRUE)

#Dodatno sredivanje tokena
tokens.test <- tokens.test %>%
  tokens_tolower() %>%
  tokens_remove(stopwords("SMART")) %>%
  tokens_remove(stopwords.tweet) %>%
  tokens_keep(min_nchar = 2) %>%
  tokens_wordstem(language = "english")

#Kreiranje DFM (document-feature matrix)
dfm.train <- dfm(tokens.train)
dfm.test <- dfm(tokens.test)

head(dfm.train)

#Kreiranje TF-IDT (term-frequency inverse document-frequency)
tfidf.train <- dfm_tfidf(dfm.train)
tfidf.test <- dfm_tfidf(dfm.test)

#Kreiranje varijable sa frekvencijama
tdm.tweet  <- TermDocumentMatrix(corpus.train)
tdm.sparse <- removeSparseTerms(tdm.tweet, 0.99)
inspect(tdm.sparse)
tdm.tweet <- as.matrix(tdm.sparse)
tdm.freq <- sort(rowSums(tdm.tweet), decreasing = TRUE)
tdm.freq <- data.frame(word = names(tdm.freq), freq = tdm.freq)

#Prikaz 20 najfrekventnijih engrama u korpusu
ggplot(tdm.freq[1:20,], aes(x=reorder(word, freq), y=freq)) + 
  geom_bar(stat="identity", fill = "#de5833") +
  xlab("Rec") + 
  ylab("Broj pojavljivanja u korpusu") + 
  coord_flip() +
  theme(axis.text=element_text(size=7)) +
  ggtitle("Najfrekventniji engrami u korpusu") +
  ggeasy::easy_center_title()

#Izdvajanje bigrama
bigram.tweet <- tibble(txt = traindata$OriginalTweet) 
bigram.tweet <- bigram.tweet %>% 
  unnest_tokens(ngram, txt, token = "ngrams", n = 2)
bigram.tweet <- as.data.frame(bigram.tweet)

#Dodatne stop reci
stopwords.bigram <- c("trending", "new")
stopwords.df <- tibble(
  word = c(stopwords(kind = "english"),
           stopwords.tweet, stopwords.bigram))

#Razdvajanje bigrama u dve kolone i uklanjanje stop reci
bigram.tweet <- bigram.tweet %>% 
  separate(col = ngram, into = c("rec1", "rec2"), sep = " ") %>% 
  filter(! rec1 %in% stopwords.df$word) %>% 
  filter(! rec2 %in% stopwords.df$word) %>% 
  filter(! is.na(rec1)) %>% 
  filter(! is.na(rec2))

#Grupisanje i prebrojavanje bigrama
bigram.count <- bigram.tweet %>% 
  dplyr::count(rec1, rec2, sort = TRUE) %>% 
  dplyr::rename(weight = n)

bigram.count %>% head(20)

#Prikaz distribucije bigrama u tekstu
bigram.count %>% 
  mutate(weight = log(weight + 1)) %>% 
  ggplot(mapping = aes(x = weight)) +
  theme_light() +
  geom_histogram() +
  labs(title = "Ponderisana distribucija bigrama u korpusu") +
  ggeasy::easy_center_title()

#Prikaz mreze bigrama
threshold <- 20
skalar.faktor <- function(x, lambda) {
  x / lambda
}

bigram.mreza <-  bigram.count %>%
  filter(weight > threshold) %>%
  mutate(weight = skalar.faktor(x = weight, lambda = 2E3)) %>% 
  graph_from_data_frame

plot(
  bigram.mreza, 
  vertex.size = 1,
  vertex.color = "light blue",
  vertex.label.color = "black", 
  vertex.label.cex = 0.6, 
  vertex.label.dist = 2
  ,
  edge.color = "gray", 
  main = "Mreza bigrama u korpusu", 
  sub = glue("Frekvencija bigrama: {threshold}"), 
  alpha = 20
)

#Kreiranje modela
#Naive Bayes - DFM
y <- traindata$Sentiment
x <- dfm.train

naive.bayes <- textmodel_nb(x = x, y = y) 

summary(naive.bayes)

#Uskladjivanje dimenzija test i train matrica
match.dfm <- dfm_match(dfm.test, 
                       features = featnames(dfm.train))

#Predvidjanje
predict.sentnb <- predict(naive.bayes, newdata = match.dfm)

#Procenjivanje modela
table.sentnb <- table(predict.sentnb, testdata$Sentiment)
table.sentnb
conf.matnb <- confusionMatrix(table.sentnb, mode = "everything")
conf.matnb

compute.eval.metrics

table(testdata$Sentiment) %>% prop.table()

#Naive Bayes - TFIDF
y <- traindata$Sentiment
x <- tfidf.train

naive.bayes1 <- textmodel_nb(x = x, y = y)

summary(naive.bayes1)

#Uskladjivanje dimenzija test i train matrica
match.tfidf <- dfm_match(tfidf.test, 
                       features = featnames(tfidf.train))

#Predvidjanje
predict.sentnb1 <- predict(naive.bayes1, newdata = match.tfidf)

#Procenjivanje modela
table.sentnb1 <- table(predict.sentnb1, testdata$Sentiment)
table.sentnb1
conf.matnb1 <- confusionMatrix(table.sentnb1, mode = "everything")
conf.matnb1

#Kreiranje modela SVM - DFM
svm1 <- textmodel_svm(x = dfm.train,
                      y = traindata$Sentiment,
                      weight = "uniform")

#Kreiranje modela SVM - TFIDF
svm.tfidf1 <- textmodel_svm(x = tfidf.train,
                            y = traindata$Sentiment,
                            weight = "uniform")

#Predvidjanje
predict.sentsvm <- predict(svm1, newdata = match.dfm)

#Procenjivanje modela
table.sentsvm <- table(predict.sentsvm, testdata$Sentiment)
table.sentsvm
conf.matsvm <- confusionMatrix(table.sentsvm, mode = "everything")
conf.matsvm

#Predvidjanje
predict.sentsvm1 <- predict(svm.tfidf1, newdata = match.dfm)

#Procenjivanje modela
table.sentsvm1 <- table(predict.sentsvm1, testdata$Sentiment)
table.sentsvm1
conf.matsvm1 <- confusionMatrix(table.sentsvm1, mode = "everything")
conf.matsvm1


