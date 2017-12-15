require(nnet)

mse = function(sm)
    mean(sm$residuals^2)

# Read in results
results = read.csv("~/git/cs221-project/src/wine_cleaned_google-final.csv", header = TRUE, sep = ",", na.strings = c(""))
results_backup = results

# Remove wine reviews without a country
country = results[,2]
results = results[!is.na(country),]

# Remove wine reviews without a varietal
variety = results[,13]
results = results[!is.na(variety),]

# Remove wine reviews without a price
price = results[,6]
results = results[!is.na(as.numeric(as.character(price))),]

# Remove wine reviews without a score
score = results[,5]
results = results[!is.na(as.numeric(as.character(score))),]

# Use these to run models
sentiment = as.numeric(results[,15])
score = as.numeric(results[,5])
price = as.numeric(results[,6])
variety = results[,13]
country = results[,2]

# Logit models
# model <- glm(sentiment ~ variety,family=binomial(link='logit'))





# Linear models
score_v_sentiment = lm(score ~ sentiment)
summary(score_v_sentiment)
mse(score_v_sentiment)

price_v_sentiment = lm(price ~ sentiment)
summary(price_v_sentiment)
mse(price_v_sentiment)

score_v_sentiment_and_price = lm(score ~ sentiment + price)
summary(score_v_sentiment_and_price)
mse(score_v_sentiment_and_price)

# Explore MSE for wines priced from [0, x]
mse_price_price = numeric()
mse_price_sentiment = numeric()
mse_price_sentiment_and_price = numeric()

# Explore MSE for wines scored from [80, x]
mse_score_price = numeric()
mse_score_sentiment = numeric()
mse_score_sentiment_and_price = numeric()

for (i in 2:100) {
    indices = price <= i
    lm = lm(score[indices] ~ price[indices])
    mse_price_price[[i]] = mse(lm)
    
    lm = lm(score[indices] ~ sentiment[indices])
    mse_price_sentiment[[i]] = mse(lm)
    
    lm = lm(score[indices] ~ sentiment[indices] + price[indices])
    mse_price_sentiment_and_price[[i]] = mse(lm)
    
    if (i >= 80) {
        indices = score <= i
        lm = lm(score[indices] ~ price[indices])
        mse_score_price[[i]] = mse(lm)
        
        lm = lm(score[indices] ~ sentiment[indices])
        mse_score_sentiment[[i]] = mse(lm)
        
        lm = lm(score[indices] ~ sentiment[indices] + price[indices])
        mse_score_sentiment_and_price[[i]] = mse(lm)
    }
}

plot(mse_price_sentiment_and_price,
     type = "l",
     xlab = "x = Price ($)",
     ylab = "Mean Squared Error",
     main = "MSE for Wines with Prices [0,x]",
     sub = "Score ~ Sentiment + Price")

plot(mse_score_sentiment_and_price,
     type = "l",
     xlim = range(80:100),
     xlab = "x = Price ($)",
     ylab = "Mean Squared Error",
     main = "MSE for Wines with Score [80,x]",
     sub = "Score ~ Sentiment + Price")

for (i in 2:100) {
    indices = price <= i
    lm = lm(score[indices] ~ price[indices])
    mse_price_price[[i]] = mse(lm)
    
    lm = lm(score[indices] ~ sentiment[indices])
    mse_price_sentiment[[i]] = mse(lm)
    
    lm = lm(score[indices] ~ sentiment[indices] + price[indices])
    mse_price_sentiment_and_price[[i]] = mse(lm)
    
    if (i >= 80) {
        indices = score <= i
        lm = lm(score[indices] ~ price[indices])
        mse_score_price[[i]] = mse(lm)
        
        lm = lm(score[indices] ~ sentiment[indices])
        mse_score_sentiment[[i]] = mse(lm)
        
        lm = lm(score[indices] ~ sentiment[indices] + price[indices])
        mse_score_sentiment_and_price[[i]] = mse(lm)
    }
}


