# sarima-model-from-scratch
 SARIMA model from scratch on monthly precipitation.
 Takes the total monthly precipitation of Brazil for 10 years to make predictions. The RMSE is about 18% which is on the higher side, but for just a SARIMA model only with precipitation amounts withoout any other additional information like humidity, it's fine. 

 ## How it works
 How it works is that it combines the AR (autoregression) and MA (moving average) with differencing and seasonal differencing to predict the next value. By differencing the data, it removes the trends and seasonal trends. Then, the AR model takes the previous values to make a prediction. The MA model also makes a prediction, but that's based on the previous prediction's error. These predictions are added together to be a final prediction. Basically, SARIMA is differencing to remove trends and seasonal trends, then the AR is the base and the MA corrects the base. Finally, after getting the prediction, you undo the differencing and transformations. The models are trained with 1000 iterations using gradient descent to optimize the coefficients.

## Use
Just clone the repository and run sarima.py
