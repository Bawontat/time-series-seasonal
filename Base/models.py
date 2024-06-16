import numpy as np
import pandas as pd
import datetime
from sklearn.linear_model import LinearRegression
import seaborn as sns
import matplotlib.pyplot as plt
from statsmodels.tsa.deterministic import DeterministicProcess,CalendarFourier
import calendar
import statistics
from texttable import Texttable

class preprocessing:
    def __init__(self,raw_data,freq_map,date_data):
        self.raw_data = raw_data
        self.freq_map = freq_map
        self.date_data = date_data

    #Split Year,Month,Date into Tuple
    def splitdate(self):
        self.size = len(self.date_data)
        self.year = []
        self.month = []
        self.date = []
        for i in range(self.size):
            timestamp = np.array(self.date_data.iloc[i].split("-") , dtype=int)
            self.year.append(timestamp[0])
            self.month.append(timestamp[1])
            self.date.append(timestamp[2])


    def preprocessing_call(self):

        #Get Tuple splited and collect into Pandas Columns
        self.splitdate()
        temp_zip = list(zip(self.date_data, self.year, self.month, self.date))
        self.date_data = pd.DataFrame(temp_zip, columns=['Timestamp', 'Year', 'Month', 'Date'])

        #------ Preprocessing Date Data ----------------

        #Get User Time period requirement and make pandas columns (Sequenctial)
        #Day of week
        if "dow" in self.freq_map:
            self.date_data["Dayofweek"] = [ datetime.date(self.year[i],self.month[i],self.date[i]).weekday()+1 for i in range(self.size)]

        #Week of month
        if "wom" in self.freq_map:
            #Set Initial State
            previousstate_month = self.date_data['Month'][0]
            #Get Ordinal number Day of months and Grouping Tuple following Ordinal number of week
            state_calendar = calendar.monthcalendar(self.date_data['Year'][0],self.date_data['Month'][0]) 
            #Preparing Constant matrix following ordinal number of week 
            week_month_matrix_mapping = np.arange(len(state_calendar)) + 1
            week_of_month = []
            for i in range(self.size):
                state_month = self.date_data['Month'][i]
                #Found Changing of month,we update state of variable
                if state_month != previousstate_month:
                    state_calendar = calendar.monthcalendar(self.date_data['Year'][i],self.date_data['Month'][i])
                    previousstate_month = self.date_data['Month'][i]
                    week_month_matrix_mapping = np.arange(len(state_calendar))+1

                #Check position of Date in Months and return 1 in matrix
                pos_incalendar = np.where(state_calendar == self.date_data['Date'][i],1,0)
                #Get position Week of months by Matrix multiplication with Constant matrix
                x = np.matmul(pos_incalendar, [1,1,1,1,1,1,1])
                #Inner Product for get Ordinal number of Week months by Matrix multiplication with Constant matrix
                week_of_month.append(np.inner(week_month_matrix_mapping , x))
            self.date_data['Weekofmonth'] = week_of_month

        #Month of year
        if "moy" in self.freq_map:
            self.date_data['Monthofyear'] = self.date_data['Month']     

        #Day of month
        if "dom" in self.freq_map:
            self.date_data['Dayofmonth'] = self.date_data['Date']

        #Week of year
        if "woy" in self.freq_map:
            #Set Initial State
            previousstate_year = self.date_data['Year'][0]
            previousstate_month = self.date_data['Month'][0]
            state_year = self.date_data['Year'][0]
            state_month = self.date_data['Month'][0]
            week_of_year = []
            stack = []

            #Initialize First Rows of Dataset to Get previous total week before month them 
            # Ex. Start at 9th month >> Get Total Week from 1st to 8th months inti previous_total_week
            for i in range(state_month-1):
                state_calendar = calendar.monthcalendar(state_year,i+1)
                stack.append(len(state_calendar))
            previous_total_week = sum(stack)

            #processing weeks of year
            for i in range(self.size):
                state_year = self.date_data['Year'][i]
                state_month = self.date_data['Month'][i]
                #Found Changing month
                if state_month != previousstate_month:
                    stack = []
                    #Month != January
                    if state_month != 1:
                        for j in range(state_month-1):
                            state_calendar = calendar.monthcalendar(state_year,j+1)
                            stack.append(len(state_calendar))
                        #Update 
                        previous_total_week = sum(stack)
                    else:
                        previous_total_week = 0 

                #Write Oridnal number Week of months
                week_of_year.append(previous_total_week + self.date_data['Weekofmonth'][i])

            self.date_data['Weekofyear'] = week_of_year

        #------ Preprocessing Raw Data to Dataset Train ----------------
        self.dataset_train  =  self.raw_data.to_frame()
        self.dataset_train = self.dataset_train.rename(columns= {self.dataset_train.columns[0]: 'TF_day'})

        #Make Time Frame : Week data & Month data by " Averge Timeframe day "
        #Setting for Week Timeframe
        previous_state_week = self.date_data['Weekofmonth'][0]
        pos_init_week = 0
        stack_week = []
        #Setting for Month Timeframe
        previous_state_month = self.date_data['Monthofyear'][0]
        pos_init_month = 0
        stack_month = []
        for i in range(self.size):
            state_week = self.date_data['Weekofmonth'][i]
            state_month = self.date_data['Monthofyear'][i]
            #Time frame : Week
            if state_week != previous_state_week:
                average_week = statistics.mean(self.dataset_train['TF_day'][pos_init_week:i])
                for j in range(i-pos_init_week):stack_week.append(average_week) 
                pos_init_week = i
                previous_state_week = self.date_data['Weekofmonth'][i]

            #Time frame : Month
            if state_month != previous_state_month:
                average_month = statistics.mean(self.dataset_train['TF_day'][pos_init_month:i])
                for j in range(i-pos_init_month):stack_month.append(average_month) 
                pos_init_month = i
                previous_state_month = self.date_data['Monthofyear'][i]

            #If The last epochs states use this step
            if i == (self.size-1):
                #Time frame : Week
                average_week = statistics.mean(self.dataset_train['TF_day'][pos_init_week:self.size])
                for j in range(self.size-pos_init_week):stack_week.append(average_week) 
                #Time frame : Month
                average_month = statistics.mean(self.dataset_train['TF_day'][pos_init_month:self.size])
                for j in range(self.size-pos_init_month):stack_month.append(average_month)                


        self.dataset_train['TF_week'] = stack_week
        self.dataset_train['TF_month'] = stack_month   

        #Insert Timestamp to dataset_train
        self.dataset_train['Timestamp'] = self.date_data['Timestamp']

    
class periodfinding(preprocessing):
    def __init__(self,raw_data,freq_map,date_data):
        preprocessing.__init__(self,raw_data,freq_map,date_data)
        self.variance_data = pd.DataFrame(columns=['Dayofweek', 'Weekofmonth', 'Monthofyear' , 'Dayofmonth' , 'Weekofyear'])

    def periodfinding_call(self):
        self.preprocessing_call()
        """
        Calculate "Day of Week" Variance by using TF_Day           > 7 Day 
                  "Week of Month" Variance by using TF_Week        > 5 Week
                  "Month of Year" Variance by using TF_Month       > 12 Month
                  "Day of Month" Variance by using TF_Day          > 31 Day
                  "Week of Year" Variance by using TF_Week         > 52 Week
        """

        # ----- Day of Week Variance ------- # Co-Factor for positioning : Week of Month
        #Initialize
        previous_weekmonth = self.date_data['Weekofmonth'][0]
        index_start = 0
        index_end = 0
        var_dow = []

        for i in range(self.size):
            state_weekmonth = self.date_data['Weekofmonth'][i]
            if state_weekmonth == previous_weekmonth:
                index_end = i
            else:
                #Some weekmonth  may be have 1 days such as start month (1) with Saturday
                if index_start != index_end:
                    cal_variance = self.dataset_train['TF_day'][index_start:(index_end+1)].var(0)
                    var_dow.append(cal_variance)
                else:
                    var_dow.append(0)

                index_start = i
                index_end = i
                previous_weekmonth = state_weekmonth
            
        # ----- [Week of month && Day of month] Variance ------- # Co-Factor for positioning : Month of Years
        #Initialize state
        previous_monthyear = self.date_data['Monthofyear'][0]
        index_start = 0
        index_end = 0
        var_wom = []
        var_dom = []

        for i in range(self.size):
            state_monthyear = self.date_data['Monthofyear'][i]
            if state_monthyear == previous_monthyear:
                index_end = i
            else:
                if index_start != index_end:
                    cal_variance_dom = self.dataset_train['TF_day'][index_start:(index_end+1)].var(0)
                    cal_variance_wom = pd.unique(self.dataset_train['TF_week'][index_start:(index_end+1)]).var(0)
                    var_dom.append(cal_variance_dom)
                    var_wom.append(cal_variance_wom)
                else:
                    var_dom.append(0)
                    var_wom.append(0)

                index_start = i
                index_end = i
                previous_monthyear = state_monthyear

        # ----- [Week of year && Month of year] Variance ------- # Co-Factor for positioning : Years
        previous_year = self.date_data['Year'][0]
        index_start = 0
        index_end = 0
        var_woy = []
        var_moy = []

        for i in range(self.size):
            state_year = self.date_data['Year'][i]
            if state_year == previous_year:
                index_end = i
            else:
                if index_start != index_end:
                    cal_variance_woy = pd.unique(self.dataset_train['TF_week'][index_start:(index_end+1)]).var(0)
                    cal_variance_moy = pd.unique(self.dataset_train['TF_month'][index_start:(index_end+1)]).var(0)
                    var_woy.append(cal_variance_woy)
                    var_moy.append(cal_variance_moy)
                else:
                    var_woy.append(0)
                    var_moy.append(0)    

                index_start = i
                index_end = i
                previous_year = state_year

        #Convert to Pandas and Save to Variance data
        self.variance_data['Dayofweek'] = pd.Series(var_dow)
        self.variance_data['Weekofmonth'] = pd.Series(var_wom)
        self.variance_data['Monthofyear'] = pd.Series(var_moy)
        self.variance_data['Dayofmonth'] = pd.Series(var_dom)
        self.variance_data['Weekofyear'] = pd.Series(var_moy)
    
        """
        Period Ranking : Get Oridinal Ranking 
        Method : Average Variance of each period and comparison
        """
        stack = []
        cols = self.variance_data.columns
        stack = [statistics.mean(self.variance_data[cols[i]].dropna()) for i in range(len(cols))]
        stack_rank = sorted(stack)
        self.period_ranking = [stack_rank.index(stack[i]) for i in range(len(stack))]
        

class frequency_analysis:
    def paramconfig_model(self):
        #Check users selection  and Set config Before start the model
        if self.period_selected_name == 'Dayofweek': 
            self.calendar_freq = 'W'
            self.calendar_order = 1
            self.walkthrough_period = "D"
            self.timeframe = 'TF_day'

        elif self.period_selected_name == 'Weekofmonth':
            self.calendar_freq = 'ME'
            self.calendar_order = 1
            self.walkthrough_period = "W"
            self.timeframe = 'TF_week'

        elif self.period_selected_name == 'Monthofyear':
            self.calendar_freq = 'YE'
            self.calendar_order = 1
            self.walkthrough_period = "ME"
            self.timeframe = 'TF_month'

        elif self.period_selected_name == 'Dayofmonth': 
            self.calendar_freq = 'ME'
            self.calendar_order = 1
            self.walkthrough_period = "D"
            self.timeframe = 'TF_day'

        elif self.period_selected_name == 'Weekofyear':
            self.calendar_freq = 'YE'
            self.calendar_order = 1
            self.walkthrough_period = "W"
            self.timeframe = 'TF_week'

    def deterministic_generator(self,start,end,fourier):
        #Set Date Range vairiable for feed into Deterministic process
        index = pd.date_range(start = start,end = end,freq=self.walkthrough_period)

        #Use Deterministic library for create Model input to Frequency analysis (Seasonal)
        dp = DeterministicProcess(
            index=index,
            constant=True,
            order=1,
            seasonal=True,
            additional_terms=[fourier],
            drop=True,)
        
        #Preprocessing Y output following Date Range variable [Same Dimensional]
        #Ex. 1st Week  of 9th month have TF Week = [99,99,99,99,99,99,99] >> So,This Below We get only one(99) following Inner join method
        key_index = pd.DataFrame(index.strftime('%Y-%m-%d'))
        key_index = key_index.rename(columns={0:'Timestamp'})

        #Return Fourier input dataset and key_index for mapping
        return dp.in_sample(),key_index


    def modelling(self):
        #Set config Before start the model
        self.paramconfig_model()

        #Set Fourier analysis for analyze Frequency Domain
        fourier = CalendarFourier(freq=self.calendar_freq, order=self.calendar_order)

        #Generate fourier dataset by deterministic library &&Key_index for mapping calue Smaller Timeframe to Higher Timeframe
        self.input_model_train,key_index_train = self.deterministic_generator(start=self.date_data['Timestamp'].iloc[0],end=self.date_data['Timestamp'].iloc[-1],fourier=fourier)

        #Inner Join Dataset_train and Date range variable
        self.output_data_train = pd.merge(key_index_train,self.dataset_train,on='Timestamp',how='inner')
        self.output_data_train = self.output_data_train.set_index('Timestamp')
        self.output_data_train = self.output_data_train[self.timeframe]

        #Fit The models
        self.model_ml = LinearRegression().fit(self.input_model_train, self.output_data_train)


    def frequency_call(self):
        #Get the name Befor modeling
        idx = [i for i in range(len(self.period_ranking)) if int(self.period_ranking[i]) == int(self.period_selected)]
        self.period_selected_name = self.variance_data.columns[idx[0]]
        print("Period Selected Name : ",self.period_selected_name)
        self.modelling()

              

class prediction(frequency_analysis):
    def predict(self):
        self.frequency_call()
        #Only the Fit Class : Test prediction by Train Dataset
        self.predict_train = pd.Series(self.model_ml.predict(self.input_model_train),index=self.input_model_train.index)


#Main Class ---------------------------------------------------------------------------
class fit(periodfinding,prediction):
    def __init__(self,raw_data,freq_map,date_data):
        periodfinding.__init__(self,raw_data,freq_map,date_data)

    def processing(self):
        self.periodfinding_call()

        #Get Input keyboard from user for select period time
        table = Texttable()
        col = self.variance_data.columns
        table.header([col[0],col[1],col[2],col[3],col[4]])
        row = self.period_ranking
        table.add_row([row[0],row[1],row[2],row[3],row[4]])
        print(table.draw())
        self.period_selected = input("Please select rank for analysis : ")
        
        self.predict()



#Main Class ---------------------------------------------------------------------------
class fit_eval(fit):
    def __init__(self,raw_data,freq_map,date_data,eval_step):
        fit.__init__(self,raw_data,freq_map,date_data)
        #Initialize Fit eval variable
        self.eval_step = eval_step
        self.dataset_eval = self.date_data_eval = 0.0
    
    #Polymorphism and Modified Method of 'Preprocessing' class
    def preprocessing_call(self):
        super().preprocessing_call()
        self.date_data_to_deterministic = self.date_data
        #Update Self.size after split Train and Eval set
        self.size = len(self.date_data)

    #Polymorphism and Modified Method deterministic_generator for Fit-Eval
    def deterministic_generator(self,start,end,fourier):
        input_model_train,key_index_train = super().deterministic_generator(start,end,fourier)

        input_model_eval = input_model_train.iloc[-self.eval_step:]
        key_index_eval = key_index_train.iloc[-self.eval_step:]

        input_model_train = input_model_train.iloc[:-self.eval_step]
        key_index_train = key_index_train.iloc[:-self.eval_step] 

        #Return Fourier input dataset and key_index for mapping
        return input_model_train ,key_index_train ,input_model_eval,key_index_eval


    #Polymorphism and Modified Method of 'modelling' class
    def modelling(self):
        #Set config Before start the model
        self.paramconfig_model()

        #Set Fourier analysis for analyze Frequency Domain
        fourier = CalendarFourier(freq=self.calendar_freq, order=self.calendar_order)

        #Generate fourier dataset by deterministic library &&Key_index for mapping calue Smaller Timeframe to Higher Timeframe
        self.input_model_train,key_index_train,self.input_model_eval,key_index_eval = self.deterministic_generator(start=self.date_data_to_deterministic['Timestamp'].iloc[0],
                                                                                end=self.date_data_to_deterministic['Timestamp'].iloc[-1],
                                                                                fourier=fourier)


        #Inner Join Dataset_train and Date range variable
        self.output_data_train = pd.merge(key_index_train,self.dataset_train,on='Timestamp',how='inner')
        self.output_data_train = self.output_data_train.set_index('Timestamp')
        self.output_data_train = self.output_data_train[self.timeframe]
        
        self.output_data_eval = pd.merge(key_index_eval,self.dataset_train,on='Timestamp',how='inner')
        self.output_data_eval = self.output_data_eval.set_index('Timestamp')
        self.output_data_eval = self.output_data_eval[self.timeframe]

        #Fit The models
        self.model_ml = LinearRegression().fit(self.input_model_train, self.output_data_train)
        



    #Polymorphism and Modified Method of 'Prediction' class
    def predict(self):
        self.frequency_call()
        #the Fit-Eval Class : Test prediction by Evaluation Dataset
        self.predict_train = pd.Series(self.model_ml.predict(self.input_model_train),index=self.input_model_train.index)
        self.predict_eval = pd.Series(self.model_ml.predict(self.input_model_eval),index=self.input_model_eval.index)
        



    