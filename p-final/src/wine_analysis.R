require(nnet)

### Functions
mse = function(sm)
    mean(sm$residuals^2)

calculate_mse = function(predicted, actual) {
    sq_errors = (predicted - actual)^2
    
    return(mean(sq_errors))
}

### Data Processing
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
results = results[!is.na(price),]

# Remove wine reviews without a score
points = results[,5]
results = results[!is.na(points),]

# Segment data
results_training = results[1:(length(results[,1]) / 4),]
results_test = results[(length(results[,1]) / 4):length(results[,1]),]

# Use these for linear predictors
sentiment = as.numeric(results[,15])
points = as.numeric(results[,5])
price = as.numeric(results[,6])
variety = results[,13]
country = results[,2]

# Use these to train models
sentiment_training = as.numeric(results_training[,15])
points_training = as.numeric(results_training[,5])
price_training = as.numeric(results_training[,6])
variety_training = results_training[,13]
country_training = results_training[,2]

# Use these to test models
sentiment_test = as.numeric(results_test[,15])
points_test = as.numeric(results_test[,5])
price_test = as.numeric(results_test[,6])
variety_test = results_test[,13]
country_test = results_test[,2]

### Logit models
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



### Linear models
lm = lm(points ~ price, data = results_training)
predicted_points_from_price = predict(lm, results_test)
calculate_mse(predicted_points_from_price, results_test[,'points'])

lm = lm(points ~ score, data = results_training)
predicted_points_from_sentiment = predict(lm, results_test)
calculate_mse(predicted_points_from_sentiment, results_test[,'points'])

lm = lm(points ~ score + price, data = results_training)
predicted_points_from_sentiment_and_price = predict(lm, results_test)
calculate_mse(predicted_points_from_sentiment_and_price, results_test[,'points'])


# Explore MSE for wines priced from [0, x]
mse_price_price = numeric()
mse_price_sentiment = numeric()
mse_price_sentiment_and_price = numeric()

# Explore MSE for wines scored from [80, x]
mse_score_price = numeric()
mse_score_sentiment = numeric()
mse_score_sentiment_and_price = numeric()


### Price Segmentation Analysis (Error Analysis)

for (i in 5:100) {
    indices_training = price_training <= i & price_training >= i - 5
    indices_test = price_test <= i & price_test >= i - 5
    
    indices_training = price_training <= 100 & price_training >= 60
    indices_test = price_test <= 100 & price_test >= 60
    
    lm = lm(points ~ price, data = results_training[indices_training,])
    predicted_points_from_price = predict(lm, results_test[indices_test,])
    mse_price_price[[i]] = calculate_mse(predicted_points_from_price, results_test[indices_test,'points'])

    lm = lm(points ~ score, data = results_training[indices_training,])
    predicted_points_from_sentiment = predict(lm, results_test[indices_test,])
    mse_price_sentiment[[i]] = calculate_mse(predicted_points_from_sentiment, results_test[indices_test,'points'])
    
    lm = lm(points ~ score + price, data = results_training[indices_training,])
    predicted_points_from_sentiment_and_price = predict(lm, results_test[indices_test,])
    mse_price_sentiment_and_price[[i]] = calculate_mse(predicted_points_from_sentiment_and_price, results_test[indices_test,'points'])
    
    
    if (i >= 80) {
        indices_training = points_training <= i & points_training >= i - 5
        indices_test = points_test <= i & points_test >= i - 5
        lm = lm(price ~ score, data = results_training[indices_training,])
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
     main = "MSE for Wines with Prices [x - 5,x]",
     sub = "Score ~ Sentiment + Price")

plot(mse_score_sentiment_and_price,
     type = "l",
     xlim = range(80:100),
     xlab = "x = Price ($)",
     ylab = "Mean Squared Error",
     main = "MSE for Wines with Score [80,x]",
     sub = "Score ~ Sentiment + Price")





### Segmentation based on WineFolly Pricing Segments

mse_price_band = function(lower_price, upper_price) {
    indices_training = price_training < upper_price & price_training >= lower_price
    indices_test = price_test < upper_price & price_test >= lower_price
    n = length(results_training[indices_training,1]) + length(results_test[indices_test,1])
    # print(c("Price lower bound: ", lower_price))
    # print(c("Price upper bound: ", upper_price))
    # print(c("# of Wines:", n))
    
        
    lm = lm(points ~ price, data = results_training[indices_training,])
    predicted_points_from_price = predict(lm, results_test[indices_test,])
    mse_pp = calculate_mse(predicted_points_from_price, results_test[indices_test,'points'])
    # print(c("MSE of Points from Price: ", mse_pp))
    
    lm = lm(points ~ score, data = results_training[indices_training,])
    predicted_points_from_sentiment = predict(lm, results_test[indices_test,])
    mse_ps = calculate_mse(predicted_points_from_sentiment, results_test[indices_test,'points'])
    # print(c("MSE of Points from Sentiment: ", mse_ps))
    
    lm = lm(points ~ score + price, data = results_training[indices_training,])
    predicted_points_from_sentiment_and_price = predict(lm, results_test[indices_test,])
    mse_psp = calculate_mse(predicted_points_from_sentiment_and_price, results_test[indices_test,'points'])
    # print(c("MSE of Points from Sentiment and price: ", mse_psp))
    
    return(c(lower_price, upper_price, n, mse_pp, mse_ps, mse_psp))
}

price_band_mse = data.frame(matrix(ncol = 6, nrow = 8))
colnames = c("min.price", "max.price", "n", "mse.points.v.price", "mse.points.v.sentiment", "mse.points.v.sentiment.price")
colnames(price_band_mse) = colnames
price_breaks = c(4, 10, 15, 20, 30, 50, 100, 200, 5000)

for (i in 1:8) {
    lower = price_breaks[[i]]
    upper = price_breaks[[i+1]]
    price_band_mse[i, 'min.price'] = lower
    price_band_mse[i, 'max.price'] = upper
    price_band_mse[i,] = mse_price_band(lower, upper)
}
price_band_mse


### Graphing Wine Price Distributions

hist(price,
     freq = TRUE,
     breaks = c(4, 10, 15, 20, 30, 50, 100, 200, 10000),
     xlim = c(0, 200),
     main = "Wine Price Distribution",
     sub = "Wines Separated Into Price Categories",
     xlab = "Price ($)")

hist(price,
     freq = TRUE,
     breaks = 3300,
     xlim = c(0, 200),
     main = "Wine Price Distribution",
     sub = "Bar Width = $1",
     xlab = "Price ($)")
