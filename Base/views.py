import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd

class fit:
    def __init__(self, input_model_train, output_data_train, predict_train, period_selected_name):
        self.input_model_train = input_model_train
        self.output_data_train = output_data_train
        self.predict_train = predict_train
        self.period_selected_name = period_selected_name
        self.timeframe = self.output_data_train.name

        #Transform Train Data
        self.input_model_train = pd.DataFrame(self.input_model_train.index.strftime('%Y-%m-%d'))
        self.input_model_train = self.input_model_train.squeeze()
        self.output_data_train = self.output_data_train.reset_index(drop=True)


    def scatterline_plot(self):  
        sns.scatterplot(x=self.input_model_train, y=self.output_data_train)
        plt.xlabel('Time Series > Date ; Period > '+self.timeframe)
        plt.ylabel('Closed Price '+self.timeframe)
        plt.title('Timeframe : '+self.period_selected_name)
        plt.plot(self.input_model_train ,self.predict_train.reset_index(drop=True) ,color = 'red')
        plt.show()


class fit_eval(fit):
    def __init__(self, input_model_train, output_data_train, predict_train, period_selected_name,input_model_eval,output_data_eval,predict_eval):
        fit.__init__(self,input_model_train, output_data_train, predict_train, period_selected_name)
        self.input_model_eval = input_model_eval
        self.output_data_eval = output_data_eval
        self.predict_eval = predict_eval
        #Transform Eval Data
        self.input_model_eval = pd.DataFrame(self.input_model_eval.index.strftime('%Y-%m-%d'))
        self.input_model_eval = self.input_model_eval.squeeze()
        self.output_data_eval = self.output_data_eval.reset_index(drop=True)    
        
    def scatterline_plot(self):  
        sns.scatterplot(x=self.input_model_train, y=self.output_data_train,color='blue')
        sns.scatterplot(x=self.input_model_eval, y=self.output_data_eval,color='red')
        plt.xlabel('Time Series > Date ; Period > '+self.timeframe)
        plt.ylabel('Closed Price '+self.timeframe)
        plt.title('Timeframe : '+self.period_selected_name)
        plt.plot(self.input_model_train ,self.predict_train.reset_index(drop=True) ,color = 'blue')
        plt.plot(self.input_model_eval ,self.predict_eval.reset_index(drop=True) ,color = 'red')
        plt.show()  