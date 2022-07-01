
library(ggplot2)
library(ggeasy)
library(caret)
library(dplyr)
library(data.table)
library(quanteda)
library(wordcloud)
library(tidytext)
library(tokenizers)
library(tm)
library(tidyr)
library(igraph)
library(glue)

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

#Skor senetimenata u korpusu

neutral <- length(which(data$SentimentNum == 0))
positive <- length(which(data$SentimentNum > 0))
negative <- length(which(data$SentimentNum < 0))

BrojTvitova <- c(positive,neutral,negative)
Sentiment <- c("Positive","Neutral","Negative")
output <- data.frame(Sentiment, BrojTvitova)
output$Sentiment<-factor(output$Sentiment,levels=Sentiment)

ggplot(output, aes(x=Sentiment,y=BrojTvitova))+
  geom_bar(stat = "identity", aes(fill = Sentiment))+
  ggtitle("Prikaz tipova sentimenta u 5000 tvitova") +
  scale_fill_brewer(palette="Set3")

#Kreiranje varijable sa duzinom tvitova
data$TweetLen <- count_characters(data$OriginalTweet)
View(data)

#Pregled lokacija 
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
  tokens_wordstem(language = "english")

#Kreiranje liste stop reci
stopwords.tweet <- c("covid2019", "covid_19", "covid19",
                     "coronaviruspandemic", "covid-19", "covid",
                     "corona", "coronavirus", "amp", "t.co", "https")

#Kreiranje korpusa
corpus.tweet <- Corpus(VectorSource(clean.tweet))
corpus.tweet <- tm_map(corpus.tweet, removeWords, stopwords.tweet)

#Kreiranje DTM
tdm.tweet  <- TermDocumentMatrix(corpus.tweet)
inspect(tdm.tweet)
tdm.sparse <- removeSparseTerms(tdm.tweet, 0.999)
inspect(tdm.sparse)
tdm.tweet <- as.matrix(tdm.sparse)
tdm.tweet <- sort(rowSums(tdm.tweet), decreasing = TRUE)
tdm.tweet <- data.frame(word = names(tdm.tweet), freq = tdm.tweet)

set.seed(123)

wordcloud(corpus.tweet, min.freq = 1, max.words = 100, scale = c(2.2,1),
          stopwords = TRUE ,colors=brewer.pal(8, "Accent"), random.color = T, 
          random.order = F)

#Prikaz 20 najfrekventnijih reci u korpusu
ggplot(tdm.tweet[1:20,], aes(x=reorder(word, freq), y=freq)) + 
  geom_bar(stat="identity", fill = "#de5833") +
  xlab("Rec") + 
  ylab("Broj pojavljivanja u korpusu") + 
  coord_flip() +
  theme(axis.text=element_text(size=7)) +
  ggtitle("Najfrekventnije reci u korpusu") +
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
  vertex.color = "lightblue",
  vertex.label.color = "black", 
  vertex.label.cex = 0.6, 
  vertex.label.dist = 2,
  edge.color = "gray", 
  main = "Mreza bigrama u korpusu", 
  sub = glue("Weight Threshold: {threshold}"), 
  alpha = 20
)