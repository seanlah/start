---
title: "MovieLens - Netflix Challenge"
author: "Aik Hean Lim"
date: "12/19/2019"
output: html_document
---
  
  ```{r setup, eval = FALSE}
knitr::opts_knit$set(root.dir = "~/Users/User/Documents/")
```

## Introduction

Going through the Netflix's massive movie library aimlessly does not seem efficient. Therefore, it is a good idea for Netflix to have a good movie recommendation system. In their goal to improve their movie recommendation by 10%, Netflix made a call-out to the data science community to come up with a model to accurately predict viewer ratings.

Netflix's users are given the choice to rate their viewing experience from 0 to 5, 0 being the worse and 5 the best. The movie reommendation system was meant to recommend movies to users according to their preference.

The Netflix challenge selects their winner by comparing the residual mean square error (RMSE) of the prediction model for movie ratings. RMSE is common method to measure typical error loss, in order to obtain the desirable outcome. Hence, the most accurate prediction model should have the lowest RMSE when computing against the actual rating scores.

The main objective of this project is to generate a predicting model similar to the models from the Netflix challenge.Accurately predicting movie ratings can be a complex task as there are many biases to be consideredn into the model. 


## Data Wrangling

### Import and Tidy Data

The original Netflix movie database used in the Netflix's challenge was not made available for public access. However, it is possible to obtain a similar dataset from the courtesy of GroupLens research lab. This database contains up to 27,000 movies, with over 20 million ratings by more than 138,000 users.

To get started, the dataset mentioned in the paragraph above can be downloaded from the follwoing url:

[link](http://files.grouplens.org/datasets/movielens/ml-10m.zip)

The 10M MovieLens version is ideal for this analysis and we can download it with the following code:

```{r import-clean, eval=TRUE, error=FALSE, message=FALSE, warning=FALSE, cache=TRUE}
if(!require(tidyverse))
  install.packages("tidyverse", repos = "http://cran.us.r-project.org")
if(!require(caret)) 
  install.packages("caret", repos = "http://cran.us.r-project.org")
if(!require(data.table)) 
  install.packages("data.table", repos = "http://cran.us.r-project.org")
if(!require(devtools))
  install.packages("devtools", repos = "http://cran.us.r-project.org")
if(!require(rafalib)) 
  install.packages("rafalib", repos = "http://cran.us.r-project.org")

# Download the 10M MovieLens dataset:
dl <- tempfile()
download.file("http://files.grouplens.org/datasets/movielens/ml-10m.zip", dl)

# Cleaning the dataset:
ratings <- fread(text = gsub("::", "\t", readLines(unzip(dl, "ml-10M100K/ratings.dat"))),
                 col.names = c("userId", "movieId", "rating", "timestamp"))
movies <- str_split_fixed(readLines(unzip(dl, "ml-10M100K/movies.dat")), "\\::", 3)
colnames(movies) <- c("movieId", "title", "genres")
movies <- as.data.frame(movies) %>% mutate(movieId = as.numeric(levels(movieId))[movieId], title = as.character(title), genres = as.character(genres))
movielens <- left_join(ratings, movies, by = "movieId")
```

### Generate Validation and Train Set

The first step after data wrangling is to develop the prediction algorithm using the test set from the larger train set. Codes to generate train and test set can be found below:

```{r validation & train set, echo=FALSE, cache = TRUE}
set.seed(1, sample.kind="Rounding")
# if using R 3.5 or earlier, use `set.seed(1)` instead
test_index <- createDataPartition(y = movielens$rating, times = 1, p = 0.1, list = FALSE)
mv_train <- movielens[-test_index,]
temp <- movielens[test_index,]

mv_test <- temp %>% 
  semi_join(mv_train, by = "movieId") %>%
  semi_join(mv_train, by = "userId")
removed <- anti_join(temp, mv_test)
mv_train <- rbind(mv_train, removed)
rm(dl, ratings, movies, temp, removed)
```
The final dataset `movielens` contain 10 million ratings (rows) with 6 variables (column).

```{r, echo = TRUE}
dim(mv_train)
glimpse(mv_train)
head(mv_train)
```

## The Dataset

The original 10M MovieLens dataset were partitioned into train and validation sets containing approximately 9 million and 1 million ratings respectively.

```{r mv_train, echo = FALSE, warning = FALSE, message = FALSE}
mv_train %>% as_tibble()
```

```{r mv_test, echo = FALSE, warning = FALSE, message = FALSE}
mv_test %>% as_tibble()
```


## Data Exploration

Every observation in the datasets indicate a rating entry by a user who rated a particular movie. However in reality, not every user rates every movie they watched. The number of distinct users and movies in the train set can be summarized with the following code:

```{r distinct-users-movies-ratings, echo=TRUE, message=FALSE, warning=FALSE}
mv_train %>% 
  summarize(users_count = n_distinct(userId),
            movies_count = n_distinct(movieId),
            min_rating = min(rating),
            max_rating = max(rating))
```
Asssuming each user watches and rates every movie known in the dataset, then there would be over 700 millions ratings. The train set on the other hand, have about 10 millions observations. This disparity gives us the idea that thematrix might actually lack density.

To better visualize the matrix, a 100 $\times$ 100 matrix with random sampling will be generated by running the code below:

```{r rating-matrix, echo=TRUE}
users_samp <- sample(unique(mv_train$userId), 100)
rafalib::mypar()
mv_train %>% filter(userId %in% users) %>% 
  select(userId, movieId, rating) %>%
  mutate(rating = 1) %>%
  spread(movieId, rating) %>% select(sample(ncol(.), 100)) %>% 
  as.matrix() %>% t(.) %>%
  image(1:100, 1:100,. , xlab="Movies", ylab="Users")
abline(h=0:100+0.5, v=0:100+0.5, col = "grey")
```

Note that tiny blocks in color within the image matrix above represents a movie rating from a user, whereas white space which fills the rest of the matrix represents otherwise. From this, we concluded that the movie-user rating matrix is indeed sparse.


## Distribution of Variables

To view how the users and movies are distributed across the dataset, frequency plot can be used to visualize the distribution:

```{r rated-movie, echo=TRUE, message=FALSE, warning=FALSE}
mv_train %>% 
  dplyr::count(movieId) %>% 
  ggplot(aes(n)) + 
  geom_histogram(bins = 20, color = "green") + 
  scale_x_log10() + 
  ggtitle("Movies") 
```

Movies with higher count are most likely blockbuster movies meanwhile those with lower count are most likely indie movies which are not known to many. 

```{r active-user, echo=TRUE, message=FALSE, warning=FALSE}
mv_train %>%
  dplyr::count(userId) %>% 
  ggplot(aes(n)) + 
  geom_histogram(bins = 30, color = "red") + 
  scale_x_log10() +
  ggtitle("Users")
```

A glimpse of the user's behaviour shows some users habitually rate every movie they watched, while some others rarely rate every movie they watched. 


## Data Analysis

It is important to know that when predicting the rating for a movie *i* by user *u*, there is a noticeable pattern such as movie *i* tends to receive certain rating and user *u* tends to give certain rating. However, groups from movies and users which are not movie *i* and user *u* tends to get different results. Therefore, the variety of information will allows to us to narrow down the choice of movies similar to movie *i* or users which have the similar rating behaviour to user *u*. In a nut shell, every information in the matrix may serve as a useful predictors for each predictions.

### Understanding the Loss Function

In short, the RMSE or residual mean squared error is the measure of accuracy for prediction algorithm. By taking $y_{u,i}$ as the actual rating (maximum rating of 5) of movie *i* given by user *u* and $\hat{y}_{u,i}$ as the prediction rating, with the same movie and user.

The formula of RMSE:
  
  $$RMSE = \sqrt{\frac{1}{N}\sum_{u,i}^{} (\hat{y}_{u,i} - y_{u,i})^2}$$
  
  Notes: *N* represent the sum of the movie/user combinations.

The function to compute the RMSE of the actual sets and their corresponding predictors is written like this:
  
  ``` {r rmse function, echo = FALSE, warning = FALSE, message = FALSE}
RMSE <- function(true_ratings, pred_ratings){
  sqrt(mean((true_ratings - pred_ratings)^2))
}
```

### First Model: Naive model

The first approach to computing the RMSE is to create a basic naive model which assume all ratings for every movies are the same, without any biases, like this:
  
  $$Y_{u,i} = \mu + \epsilon_{u,i}$$
  
  $\mu$ is defined as the mean or "true" rating for all movies.
$\epsilon_{u,i}$ is defined as the independent error observed at $\mu$.

This naive model will serve as a reference for the following models which will included the relevant biases to improve the prediction accuracy.

To standardized the ratings, the mean rating is calculated from the rating variable:
  
  ```{r mu_hat, echo = FALSE, warning = FALSE, message = FALSE}
rating_mu_hat <- mean(mv_train$rating)
rating_mu_hat
```

The ${hat}\mu$ rating or rating estimate is about 3.5 stars.

Calculation of the naive RMSE can be done with following code:
  
  ``` {r naive_rmse, echo = FALSE, warning = FALSE, message = FALSE}
naive_rmse <- RMSE(mv_test$rating, rating_mu_hat)
naive_rmse
```

Note that the value of the naive RMSE is more than 1, indicating the error of prediction is more than one star of the actual rating (positive or negative), and thus not a reliable prediction model.

The next step is to include the biases which are relevant to the prediction model such as which movie *i* and which user rates gave the ratings *u*.


### Second Model: Movie Effect Model

In general, not all movie are the same. Movies ratings depends on various factors, such as storyline, genre, quality and so on.

By adding the movie effect $b_{i}$ into the naive model, the new model will be:
  
  $$Y_{u,i} = \mu + b_{i} + \epsilon_{u,i}$$
  
  
  Thus, in this second model bias or effect that will be consider the movie-specific effect in order to reduce the RMSE.

```{r movie_b_i, echo = FALSE, warning = FALSE, message = FALSE}
mu <- mean(mv_train$rating)
movie_b_i <- mv_train %>% group_by(movieId) %>% 
  summarize(b_i = mean(rating - mu))
movie_b_i %>% qplot(b_i, geom ="histogram", bins = 20, data = ., color = I("blue"))
```

As seen above, "good" movies usually have positive $b_{i}$, while "bad" movies have negative effect.


```{r predicted-ratings, echo = TRUE,}
pred_ratings <- mu + mv_test %>% left_join(movie_b_i, by='movieId') %>% pull(b_i)


# calculate rmse by modelling movie effect
rmse_2 <- RMSE(pred_ratings, mv_test$rating)
rmse_2
```

Besides that, a popular(high rating frequency) movie does not mean that it will surely have higher average ratings than a less popular one (low rating frequency). This will be discuss later in the analysis.


### Third Model: Movie + User Effect Model

Apart from movie effects *i*, user effect *u* is just as important and both effect should be included together in one model.

Take a closer look at the rating behaviour of the users, which is the most common rating given by the users?
  
  ``` {r user-bias, echo = FALSE, warning = FALSE, message = FALSE}
mv_train %>% group_by(userId) %>% 
  summarize(b_u = mean(rating)) %>% 
  filter(n() >= 100) %>% 
  ggplot(aes(b_u)) + 
  geom_histogram(bins = 30, color = "black")
```

Although a movie *i* was given high rating by most users, there are some users who gave averagely lower ratings compare to their high-rating-giver counterpart. This implies that user-specific effect is also a determining factor for the given movie rating. Adding user-specific effect b_{u} to the equation:
  
  $$Y_{u,i} = \mu + b_{i} + b_{u} + \epsilon_{u,i}$$
  
  Similar to movie effect, the user effect, *u*, can be computed as follows:
  
  ```{r user_avgs, echo = FALSE, warning=FALSE, message=FALSE}
user_bu_avgs <- mv_train %>% 
  left_join(movie_b_i, by='movieId') %>%
  group_by(userId) %>%
  summarize(b_u = mean(rating - mu - b_i))
```

# Computing RMSE of Movie + User effect Model:
```{r predicted-ratings2, echo = TRUE}
pred_ratings2 <- mv_test %>% 
  left_join(movie_avgs, by='movieId') %>%
  left_join(user_avgs, by='userId') %>%
  mutate(pred = mu + b_i + b_u) %>%
  pull(pred)

# calculate RMSE by modelling movie + user effect
rmse_3 <- RMSE(pred_ratings2, mv_test$rating)
rmse_3
```
Hence, the RMSE is further reduced to a lower value with the addition of the user-specific effect, $$b_{u}$$. 


## Regularization

Do all movie ratings comes from groups with adequate number of raters/viewers? How large is the sample size for user who rated the "best" and the "worse" movie?
  
  Creating the movie library:
  ```{r, include = FALSE}
all_movie_titles <- movielens %>% 
  select(movieId, title) %>%
  distinct()
```

These are amongst the 10 best and worse movies according to our users:
  ```{r, echo = TRUE}
# List of best movies and its rating frequency respectively

movie_b_i %>% left_join(all_movie_titles, by="movieId") %>%
  arrange(desc(b_i)) %>% 
  slice(1:10)  %>% 
  pull(title)

mv_train %>% count(movieId) %>% 
  left_join(movie_b_i, by="movieId") %>%
  left_join(all_movie_titles, by="movieId") %>%
  arrange(desc(b_i)) %>% 
  slice(1:10) %>% 
  pull(n)

# List of best movies and its rating frequency respectively

movie_b_i %>% left_join(all_movie_titles, by="movieId") %>%
  arrange(b_i) %>% 
  slice(1:10)  %>% 
  pull(title)

mv_train %>% count(movieId) %>% 
  left_join(movie_b_i) %>%
  left_join(all_movie_titles, by="movieId") %>%
  arrange(b_i) %>% 
  slice(1:10) %>% 
  pull(n)
```


As mentioned before, these are the movies which are rated or watched by a handful of users, in this case, mostly by just one user, whom gave the movies their best or worse possible ratings.

These are the noisy factors which can disrupt the estimates by producing large errors, therefore increasing the RMSE. This uncertainty will cause the computation to produce larger estimates of b_{i}, either being more positive or more negative.

In this situation, regularization should be used to the fit the model.

Movie + User effect Model:
  $$Y_{u,i} = \mu + b_i + + b_u + \varepsilon_{u,i}$$
  
  With the addition of new parameter for regularization:
  $$\frac{1}{N} \sum_{u,i} \left(y_{u,i} - \mu - b_i\right)^2 + \lambda \sum_{i} b_i^2$$ or
$$\frac{1}{N} \sum_{u,i} \left(y_{u,i} - \mu - b_i - b_u \right)^2 + \lambda \left(\sum_{i} b_i^2 + \sum_{u} b_u^2\right)$$
  
  In brief, regularization is a method to limit the variability by penalizing large estimates which are produced from small sample sizes. This is because estimates from groups with small sample size are known to cause high levels of uncertainty in the predictions.


### Fourth Model: Regularized Movie + User Effect Model

Regularizing movie and user effect in one model:
  
  $$\frac{1}{N} \sum_{u,i} \left(y_{u,i} - \mu - b_i - b_u \right)^2 + 
  \lambda \left(\sum_{i} b_i^2 + \sum_{u} b_u^2\right)$$
  
  Computing both effect in one model:
  ```{r lambda, echo = TRUE, message=FALSE, warning=FALSE}
lambdas <- seq(0, 10, 0.5)
rmse_f <-sapply(lambdas, function(l){
  mu <- mean(train$rating)
  b_i <- train %>% 
    group_by(movieId) %>%
    summarize(b_i = sum(rating - mu)/(n()+l))
  b_u <- train %>% 
    left_join(b_i, by="movieId") %>%
    group_by(userId) %>%
    summarize(b_u = sum(rating - b_i - mu)/(n()+l))
  predicted_ratings <- 
    validation %>% 
    left_join(b_i, by = "movieId") %>%
    left_join(b_u, by = "userId") %>%
    mutate(pred = mu + b_i + b_u) %>%
    pull(pred)
  
  return(RMSE(pred_ratings, mv_train$rating))
})
qplot(lambdas, rmse_f) 
```


Selecting the lambda value which produces the minimum value of RMSE:
  
  ``` {r optimal-lambda, echo = TRUE, message = FALSE, warning = FALSE}
lambda <- lambdas[which.min(rmses)]
lambda

# RMSE of regularized movie + user effect model
rmse_4 <- min(rmse_f)
```


## Results

By running the models step by step, the RMSE gradually decreases from the model to model. From the basic na?ve model, the computed RMSE (RMSE = 1.0612018) does not seems satisfying as this model produced an error more than a star to the prediction model. Then, the second model (movie effect model) shown a significant drop in RMSE (RMSE = 0.9439087) from the first model (naive model) by computing the movie effect bias into the model. Again, significant reduction of RMSE (RMSE = 0.8653488) seen from the third model, with the computation of movie and user effect combination. To prevent model from producing larger estimates with small sample size, the method of regularization is then used to further reduce the RMSE. The fourth model (regularized movie + user effect) produced RMSE (0.8648177).

The results can be summarized into a table below:
  
  | Model | RMSE |
  | ---------------------------------|-------------|
  | Naive Model | `naive_rmse` |
  | Movie effect Model | `rmse_2` |
  | Movie + User effect Model | `rmse_3` |
  | Regularized MOvie + User effect Model | `rmse_4` |
  
  ## Conclusion
  
  The main goal is to reduce the prediction error of the model, and obtain the lowest possible RMSE when computing model to predict movie ratings. Depending on the effect or bias, the prediction model will achieved the desire outcome.

## References
* https://bits.blogs.nytimes.com/2009/09/21/netflix-awards-1-million-prize-and-starts-a-new-contest

* http://blog.echen.me/2011/10/24/winning-the-netflix-prize-a-summary

* https://www.netflixprize.com/assets/GrandPrize2009_BPC_BellKor.pdf

* hhttps://rafalab.github.io/dsbook/large-datasets.html#regularization