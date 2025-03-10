# Stock Price Prediction with LSTM in PyTorch

This Jupyter Notebook provides step-by-step instructions to implement a stock price prediction model using **Long Short-Term Memory (LSTM)** networks, with NVIDIA (NVDA) stock data as an example. Below are the details of the project's workflow, dependencies, and instructions.

---

## **Project Workflow**

The stock price prediction task is divided into the following steps:

1. **Fetch Stock Data**: Use the `yfinance` library to download historical stock data for NVIDIA (NVDA) from January 1, 2020, to March 9, 2025. The data includes `Open`, `Close`, `High`, and `Low` prices.

2. **Visualize Stock Trends**: Create line plots for `Open` and `Close` prices over time to observe market trends.

3. **Calculate Moving Averages**: Compute 100-day, 200-day, and 300-day moving averages for analyzing stock trends and smoothing price changes.

4. **Split Data**: Divide the data into **training (80%)** and **testing (20%)** datasets for model development and evaluation.

5. **Preprocess the Data**: Scale the stock prices to the range `[0, 1]` using `MinMaxScaler` and create sliding window sequences of the past 100 days as input for the model.

6. **Define the LSTM Model**: Implement the LSTM model in `PyTorch`, which takes 100-day sequences of stock prices as input and predicts the next day's price.

7. **Convert Data to PyTorch Tensors**: Prepare the training data and convert the NumPy arrays to PyTorch tensors, optionally utilizing a GPU (if available).

8. **Train the Model**: Optimize the model using **Mean Squared Error Loss (MSE)** and the **Adam optimizer** over 20 epochs.

9. **Test the Model**: Evaluate the model on the test data, make predictions, and compare them against actual values.

10. **Visualize Results**: Plot the predicted prices against the actual prices for the test dataset to assess performance.

11. **Predict Next-Day Price**: Use the most recent 100 days of data to predict the stock's closing price for the next day.

12. **Compare Predictions**: Compare the most recent actual closing price to the predicted next day's price and calculate the percentage difference.

---

## **Dependencies**

To run this notebook, you'll need the following Python libraries:

- **Jupyter Notebook**: For interactive coding.
- **yfinance**: To download and process historical stock data.
- **numpy**: For numerical computations.
- **pandas**: For data manipulation and preprocessing.
- **matplotlib**: For data visualization.
- **scikit-learn**: For scaling stock prices using `MinMaxScaler`.
- **PyTorch**: For building and training the LSTM model.

Install these dependencies using the following command if you haven't already:

```bash
pip install jupyter yfinance numpy pandas matplotlib scikit-learn torch
```

---

## **Notebook Highlights**

- **Moving Averages**: This project emphasizes the importance of moving averages (100-day, 200-day, etc.) and their role in analyzing stock trends.
- **Deep Learning with PyTorch**: Stock price prediction is implemented using an LSTM model, which is well-suited for sequential data like time series.
- **GPU Support**: Utilizes GPU acceleration (if available) for faster training.

---

## **Usage Instructions**

1. Clone or download this repository.
   
2. Open the notebook in Jupyter Notebook:
   ```bash
   jupyter notebook stock_price_prediction.ipynb
   ```

3. Run the notebook cell-by-cell following the outlined steps in the markdown headers.

4. Once the entire notebook is executed, observe the predictions, visualizations, and comparison results at the end.

---

## **Results**

- **Please Note** that **START_DATE**, **END_DATE**, and **TICKER** can be changed to any date range or ticker preferred in **Step 1**.
- Visualizes trends of NVIDIA stock from 2020 to 2025. 
- Trains a neural network to predict stock prices based on historical data.
- Compares actual prices vs predicted prices for NVIDIA stock.
- Predicts the closing price for the next day.

---

## **Limitations**

- This notebook is designed for educational purposes and should not be used for actual trading or financial investment decisions.
- Stock prediction is inherently uncertain and influenced by many external factors not accounted for in this model.

---

## **Future Improvements**

- Incorporate external factors (e.g., news sentiment, market indexes) to enhance predictions.
- Extend the pipeline to support multiple stock tickers.

---

## **Acknowledgements**

- This project is inspired by and uses concepts from the repository:
- *Stock Price Prediction Using LSTM* by **Adarsh Yadav**
- GitHub Repo: https://github.com/034adarsh/Stock-Price-Prediction-Using-LSTM

---

## **License**

This project is licensed under the MIT License.

For feedback or questions, feel free to open an issue or contribute to this repository.