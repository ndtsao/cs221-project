require(nnet)

mse = function(sm)
    mean(sm$residuals^2)

# Read in results
results = read.csv("~/git/cs221-project/src/wine_cleaned_google-final.csv", header = TRUE, sep = ",", na.strings = c(""))
results_backup = results
results = results_backup

# Remove wine reviews without a country
country = results[,2]
results = results[!is.na(country),]

# Remove wine reviews without a varietal
variety = results[,13]
results = results[!is.na(variety),]

# Remove wine reviews without a price
price = results[,6]
# results = results[!is.na(as.numeric(as.character(price))),]
results = results[!is.na(price),]

# Remove wine reviews without a score
score = results[,5]
# results = results[!is.na(as.numeric(as.character(score))),]
results = results[!is.na(score),]

# Segment data
results_training = results[1:(length(results[,1]) / 4),]
results_test = results[(length(results[,1]) / 4):length(results[,1]),]

# Use these for linear predictors
sentiment = as.numeric(results[,15])
score = as.numeric(results[,5])
price = as.numeric(results[,6])
variety = results[,13]
country = results[,2]

# Use these to train models
sentiment_training = as.numeric(results_training[,15])
score_training = as.numeric(results_training[,5])
price_training = as.numeric(results_training[,6])
variety_training = results_training[,13]
country_training = results_training[,2]

# Use these to test models
sentiment_test = as.numeric(results_test[,15])
score_test = as.numeric(results_test[,5])
price_test = as.numeric(results_test[,6])
variety_test = results_test[,13]
country_test = results_test[,2]

# Logit models
# Train models
# country_sentiment_price_model = multinom(country_training ~ sentiment_training + price_training)
country_sentiment_price_model = multinom(country ~ score + price, data = results_training, MaxNWts = 15000)
# variety_sentiment_price_model = multinom(variety_training ~ sentiment_training + price_training)
variety_sentiment_price_model = multinom(variety ~ score + price, data = results_training, MaxNWts = 20000)
# country_sentiment_model = multinom(country_training ~ sentiment_training)
country_sentiment_model = multinom(country ~ score, data = results_training, MaxNWts = 15000)
# variety_sentiment_model = multinom(variety_training ~ sentiment_training)
variety_sentiment_model = multinom(variety ~ score, data = results_training, MaxNWts = 15000)

summary(country_sentiment_price_model)
summary(variety_sentiment_price_model)
summary(country_sentiment_model)
summary(variety_sentiment_model)

# Run logit models on test data
predicted_country_sp = predict(country_sentiment_price_model, data = results_test, "probs")
predicted_variety_sp = predict(variety_sentiment_price_model, data = results_test, "probs")
predicted_country_s = predict(country_sentiment_model, data = results_test, "probs")
predicted_variety_s = predict(country_sentiment_model, data = results_test, "probs")

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


