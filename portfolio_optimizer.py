#!/usr/bin/env python
# coding: utf-8

# ## Accessing the Stock Price Data

# In[1]:


get_ipython().system('pip install pandas-datareader')


# In[34]:


get_ipython().system('pip install yfinance')


# In[129]:


from pandas_datareader import data as pdr
import yfinance as yf
import datetime
import pandas as pd


# In[130]:


pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)


# In[131]:


start = datetime.datetime(2019,1,1)
end = datetime.datetime(2022,12,31)


# In[132]:


# fucntion that gets the stock data
def get_stock(ticker):
    yf.pdr_override()
    data = pdr.get_data_yahoo(f"{ticker}",start,end)
    data[f'{ticker}'] = data["Close"]
    data = data[[f'{ticker}']] 
    data.reset_index(inplace=True)
    data.index = data.index.astype(int)
    return data 


# In[71]:


pfizer = get_stock("PFE")
jnj = get_stock("JNJ")


# ## Stocks to be pulled are:
# #### Healthcare : Moderna (MRNA), Pfizer (PFE), Johnson & Johnson (JNJ)
# #### Tech : Google (GOOGL), Facebook/META (META), Apple (AAPL)
# #### Retail : Costco (COST), Walmart (WMT), Kroger Co (KR)
# #### Finance : JPMorgan Chase & Co (JPM), Bank of America (BAC), HSBC Holding (HSBC)

# In[133]:


from functools import reduce

def combine_stocks(tickers):
    data_frames = []
    for i in tickers:
        data_frames.append(get_stock(i))
    df_merged = reduce(lambda  left,right: pd.merge(left,right,on=['Date'], how='outer'), data_frames)
    print(df_merged.head())
    return df_merged


# In[134]:


stocks = ["MRNA", "PFE", "JNJ", "GOOGL", 
          "META", "AAPL", "COST", "WMT", "KR", "JPM", 
          "BAC", "HSBC"]
portfolio = combine_stocks(stocks)


# In[135]:


portfolio.to_csv("portfolio.csv", index=False)


# In[136]:


portfolio = pd.read_csv("portfolio.csv", parse_dates=['Date'])


# # Mean Variance Optimization

# In[63]:


get_ipython().system('pip install PyPortfolioOpt')


# In[137]:


from pypfopt.expected_returns import mean_historical_return
from pypfopt.risk_models import CovarianceShrinkage


# In[138]:


# portfolio[['MRNA', 'PFE', 'JNJ', 'GOOGL', 'META', 'APPL', 'COST', 'WMT', 'KR', 'JPM', 'BAC', 'HSBC']] = portfolio[['MRNA', 'PFE', 'JNJ', 'GOOGL', 'META', 'AAPL', 'COST', 'WMT', 'KR', 'JPM', 'BAC', 'HSBC']].astype(float)


# In[141]:


mu = mean_historical_return(portfolio[['MRNA', 'PFE', 'JNJ', 'GOOGL', 'META', 'AAPL', 'COST', 'WMT', 'KR', 'JPM', 'BAC', 'HSBC']])
S = CovarianceShrinkage(portfolio[['MRNA', 'PFE', 'JNJ', 'GOOGL', 'META', 'AAPL', 'COST', 'WMT', 'KR', 'JPM', 'BAC', 'HSBC']]).ledoit_wolf()


# In[142]:


from pypfopt.efficient_frontier import EfficientFrontier

ef = EfficientFrontier(mu,S)
weights = ef.max_sharpe()

cleaned_weights = ef.clean_weights()
print(dict(cleaned_weights))
print("-----------------------------------")
ef.portfolio_performance(verbose=True)


# #### Considering the investment amount as $100,000

# In[143]:


from pypfopt.discrete_allocation import DiscreteAllocation, get_latest_prices

latest_prices = get_latest_prices(portfolio).dropna()
latest_prices = pd.to_numeric(latest_prices, errors='coerce')
latest_prices = latest_prices.iloc[1:]  # drop the first row
# print(latest_prices)
amount = 100000
da = DiscreteAllocation(weights, latest_prices, total_portfolio_value=amount)

allocation, leftover = da.greedy_portfolio()
print("Discrete allocation:", allocation)
print("Funds remaining: ${:.2f}".format(leftover))


# ## Conclusion
# 
# We see that our portfolio performs with an expected annual return of 43 percent. The Sharpe ratio value of 1.34 indicates that the portfolio optimization algorithm performs well with our current data. Of course, this return is inflated and is not likely to hold up in the future. 
# 
# Mean variance optimization doesn’t perform very well since it makes many simplifying assumptions, such as returns being normally distributed and the need for an invertible covariance matrix. Fortunately, methods like HRP and mCVAR address these limitations. 

# # Hierarchical Risk Parity (HRP)

# In[144]:


from pypfopt import HRPOpt


# In[145]:


returns = portfolio[['MRNA', 'PFE', 'JNJ', 'GOOGL', 'META', 'APPL', 'COST', 'WMT', 'KR', 'JPM', 'BAC', 'HSBC']] = portfolio[['MRNA', 'PFE', 'JNJ', 'GOOGL', 'META', 'AAPL', 'COST', 'WMT', 'KR', 'JPM', 'BAC', 'HSBC']].pct_change().dropna()


# In[146]:


hrp = HRPOpt(returns)
hrp_weights = hrp.optimize()


# In[147]:


print(dict(hrp_weights))
print("-----------------------------------")
hrp.portfolio_performance(verbose=True)


# In[148]:


da_hrp = DiscreteAllocation(hrp_weights, latest_prices, total_portfolio_value=100000)

allocation, leftover = da_hrp.greedy_portfolio()
print("Discrete allocation (HRP):", allocation)
print("Funds remaining (HRP): ${:.2f}".format(leftover))


# # Conclusion
# 
# We see that we have an expected annual return of 15.9 percent, which is significantly less than the inflated 43.0 percent we achieved with mean variance optimization. We also see a diminished Sharpe ratio of 0.75.  This result is much more reasonable and more likely to hold up in the future since HRP is not as sensitive to outliers as mean variance optimization is. 
# 
# We see that our algorithm suggests we invest heavily into Kroger (KR), HSBC, Johnson & Johnson (JNJ) and Pfizer (PFE) and not, as the previous model did, so much into Moderna (MRNA) and Apple (AAPL). Further, while the performance decreased, we can be more confident that this model will perform just as well when we refresh our data. This is because HRP is more robust to the anomalous increase in Moderna and Apple stock prices.

# # Mean Conditional Value at Risk (mCVAR)

# In[149]:


from pypfopt.efficient_frontier import EfficientCVaR


# In[150]:


S = portfolio[['MRNA', 'PFE', 'JNJ', 'GOOGL', 'META', 'APPL', 'COST', 'WMT', 'KR', 'JPM', 'BAC', 'HSBC']].cov()
ef_cvar = EfficientCVaR(mu, S)
cvar_weights = ef_cvar.min_cvar()

cleaned_weights = ef_cvar.clean_weights()
print(dict(cvar_weights))
print("-----------------------------------")
ef_cvar.portfolio_performance(verbose=True)


# In[151]:


da_cvar = DiscreteAllocation(cvar_weights, latest_prices, total_portfolio_value=100000)

allocation, leftover = da_cvar.greedy_portfolio()
print("Discrete allocation (CVAR):", allocation)
print("Funds remaining (CVAR): ${:.2f}".format(leftover))


# # Conclusion
# 
# We see that this algorithm suggests we invest heavily into Johnson & Johnson (JNJ) and Bank of America (BAC). Also it suggests to buy a single share of HSBC. Also we see that the expected return is 12.7 percent.As with HRP, this result is much more reasonable than the inflated 43 percent returns given by mean variance optimization since it is not as sensitive to the anomalous behaviour of the Moderna stock price. 
