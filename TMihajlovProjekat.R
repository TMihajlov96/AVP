
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
#install.packages("nnet")
library(nnet)

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
describe(data, na.rm = TRUE, interp=FALSE,skew = TRUE, ranges = TRUE,trim=.1,
         type=3,check=TRUE,fast=NULL,quant=TRUE, IQR=FALSE,omit=FALSE,data=NULL) 

#Duzina tvita u odnosu na sentiment - 
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
data$TweetAt <- as.Date(data$TweetAt, format = "%d/%m/%y") #dis ok
dates <- as.data.frame(data$TweetAt) #dis ok
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

#Tokenizacija tvitova
clean.tweet <- tokens(data$OriginalTweet,
                      what = "word1",
                      remove_punct = TRUE,
                      remove_symbols = TRUE,
                      remove_numbers = TRUE,
                      remove_url = TRUE,
                      remove_twitter = TRUE)

#Dodatno sredivanje tokena
clean.tweet <- clean.tweet %>%
  tokens_tolower() %>%
  tokens_remove(stopwords("SMART")) %>%
  tokens_keep(min_nchar = 2) %>%
  tokens_wordstem(language = "english") %>%

clean.list <- as.list(clean.tweet)

#Kreiranje liste stop reci
stopwords.tweet <- c("covid2019", "covid_19", "covid19",
                     "coronaviruspandemic", "covid-19", "covid",
                     "corona", "coronavirus", "amp", "t.co", "https")

#Kreiranje korpusa
corpus.tweet <- Corpus(VectorSource(clean.list))
corpus.tweet <- tm_map(corpus.tweet, removeWords, stopwords.tweet)
corpus.tweet <- tm_map(corpus.tweet, removePunctuation)

#Kreiranje DTM
tdm.tweet  <- TermDocumentMatrix(corpus.tweet)
inspect(tdm.tweet)
tdm.sparse <- removeSparseTerms(tdm.tweet, 0.99)
inspect(tdm.sparse)
tdm.tweet <- as.matrix(tdm.sparse)
tdm.freq <- sort(rowSums(tdm.tweet), decreasing = TRUE)
tdm.freq <- data.frame(word = names(tdm.freq), freq = tdm.freq)

set.seed(123)

wordcloud(corpus.tweet, min.freq = 1, max.words = 100, scale = c(2.2,1),
          stopwords = TRUE ,colors=brewer.pal(8, "Accent"), random.color = T, 
          random.order = F)

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
bigram.tweet <- tibble(txt = data$OriginalTweet) 
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
#ovo srediti naci objasnjenje u radu tacno sta je 
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
  sub = glue("Weight Threshold: {threshold}"), 
  alpha = 20
)

#Izdvajanje podataka za model
tdm.df <- as.data.frame(t(tdm.tweet))
tdm.df <- cbind(tdm.df, data$Sentiment)
colnames(tdm.df)[221] <- "Sentiment"
as.factor(tdm.df$Sentiment)

#Kreiranje test i trening seta
train.indices <- createDataPartition(tdm.df, p=0.8, list = FALSE)
tdm.sent <- tdm.df$Sentiment
tdm.term <- tdm.df[, !colnames(tdm.df) %in% "Sentiment"]
tdm.term <- as.data.frame(tdm.term, stringsAsFactors = FALSE)
typeof(tdm.term)
View(tdm.term)
traindata <- tdm.df[train.indices,]
testdata <- tdm.df[1:220][-train.indices,]
as.factor(tdm.sent)

class(tdm.sent)
#Kreiranje modela
#Logisticka regresija model 1

lrm1 <- multinom(formula = tdm.sent ~ tdm.term, data = tdm.df)
