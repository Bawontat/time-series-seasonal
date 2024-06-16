from Base import models,views
import time

#Train Model by Train_dataset and Shown Prediction of Train_dataset
class fit:
    def __init__(self,raw_data,frequency,date_data):
        self.raw_data = raw_data
        self.date_data = date_data
        self.freq_map = frequency

    def processing(self):
        m = models.fit(self.raw_data,self.freq_map,self.date_data)
        m.processing()
        v = views.fit(m.input_model_train ,m.output_data_train ,m.predict_train ,m.period_selected_name)
        v.scatterline_plot()


#Train Model by Train_dataset(Splited out n rows data) and Shown Prediction of Train_dataset(use n rows data to predit)
class fit_eval:
    def __init__(self,raw_data,frequency,date_data,eval_step):
        self.raw_data = raw_data
        self.eval_step = eval_step
        self.date_data = date_data
        self.freq_map = frequency

    def processing(self):
        m = models.fit_eval(self.raw_data,self.freq_map,self.date_data,self.eval_step)
        m.processing()
        v = views.fit_eval(m.input_model_train ,m.output_data_train ,m.predict_train ,m.period_selected_name,m.input_model_eval,m.output_data_eval,m.predict_eval)
        v.scatterline_plot()
        