from Base import controllers
import pandas as pd

#Import Raw Data set to System
path_raw_data = "Dataset/BTC-USD.csv"
raw_data = pd.read_csv(path_raw_data)
close_data = raw_data.loc[:, 'Close']
date_data = raw_data.loc[:, 'Date']

"""

1.Day of week   : dow 
2.Week of month : wom
3.Month of year : moy
4.Day of Month  : dom
5.Week of year  : woy

"""
frequency_pos = ['dow','wom','moy','dom','woy']

"""
#print("--------------- Fit Test -------------------")
objective_fit = controllers.fit(close_data,frequency_pos,date_data)
objective_fit.processing()
"""


print("--------------- Fit_eval Test -------------------")
eval_step = 10
objective_fiteval = controllers.fit_eval(close_data,frequency_pos,date_data,eval_step)
objective_fiteval.processing()