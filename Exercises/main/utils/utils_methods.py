import numpy as np
import pandas as pd
import os
from collections import UserDict
from supersmoother import SuperSmoother, LinearSmoother
from statsmodels.tsa.seasonal import STL
from scipy.interpolate import interp1d
import matplotlib.pyplot as plt


def X_y(data, HORIZON, tensor_structure, freq = "D", variable = 'traffic'):
    #breakpoint()
    data_inputs = TimeSeriesTensor(data, variable, HORIZON, tensor_structure, freq = freq)
    DT = data_inputs.dataframe
    X = DT.iloc[:, DT.columns.get_level_values(0)=='encoder_input']
    y = DT.iloc[:, DT.columns.get_level_values(0)=='target']
    return data_inputs, X, y

def embed_data(train, valid, test, HORIZON, LAG, freq = "D", variable = 'traffic'):
    #tensor_structure = {'encoder_input':(range(-LAG+1, 1), ['traffic']), 'decoder_input':(range(0, HORIZON), ['traffic'])}
    tensor_structure = {'encoder_input':(range(-LAG+1, 1), [variable]), 'decoder_input':(range(0, HORIZON), [variable])}
    train_inputs, X_train, y_train = X_y(train, HORIZON, tensor_structure, freq = freq, variable = variable)
    valid_inputs, X_valid, y_valid = X_y(valid, HORIZON, tensor_structure, freq = freq, variable = variable)
    test_inputs, X_test, y_test   = X_y(test, HORIZON, tensor_structure, freq = freq, variable = variable)
    return train_inputs, valid_inputs, test_inputs, X_train, y_train, X_valid, y_valid, X_test, y_test

def plot_learning_curves(history):
    plot_df = pd.DataFrame.from_dict({'train_loss':history.history['loss'], 'val_loss':history.history['val_loss']})
    plot_df.plot(logy=True, figsize=(10,10), fontsize=12)
    plt.xlabel('epoch', fontsize=12)
    plt.ylabel('loss', fontsize=12)
    plt.show()

def plot_forecasts(eval_df, HORIZON, h):
    plot_df = eval_df[(eval_df.h=='t+1')][['timestamp', 'actual']]
    if HORIZON > 1:
        plot_df['t+'+str(h)] = eval_df[(eval_df.h=='t+'+str(h))]['prediction'].values
        #fig = plt.figure(figsize=(15, 8))
        plt.plot(plot_df['timestamp'], plot_df['actual'], color='red', linewidth=4.0)
        plt.plot(plot_df['timestamp'], plot_df['t+'+str(h)], color='blue', linewidth=4.0, alpha=0.75) 
    else:
        eval_df.plot(x='timestamp', y=['prediction', 'actual'], style=['r', 'b'], figsize=(15, 8))

    plt.xlabel('timestamp', fontsize=12)
    plt.ylabel('traffic', fontsize=12)
    plt.show()

def clean(series):
    n_series = len(series)
    if n_series % 2 == 0:
        n_series = n_series - 1
    stl = STL(series, period = 7, robust = True, seasonal = n_series)
    res = stl.fit()

    detrend = series - res.trend 
    strength = 1 - np.var(res.resid) / np.var(detrend)
    if strength >= 0.6:
        series = res.trend + res.resid # deseasonlized series

    tt = np.arange(len(series))
    model = SuperSmoother()
    model.fit(tt, series)
    yfit = model.predict(tt)
    resid = series - yfit

    resid_q = np.quantile(resid, [0.25, 0.75])
    iqr = np.diff(resid_q)
    #limits = resid.q + 3 * iqr * [-1, 1]
    limits = resid_q + 5 * iqr * [-1, 1]

    
    # Find residuals outside limits
    series_cleaned = series.copy()
    outliers = None
    if (limits[1] - limits[0]) > 1e-14:
        outliers = [a or b for a, b in zip((resid < limits[0]).to_numpy() , (resid > limits[1]).to_numpy())]
        if any(outliers):
            series_cleaned.loc[outliers] = np.nan
            # Replace outliers 
            id_outliers = [i for i, x in enumerate(outliers) if x]
            for ii in id_outliers:
                xx = [ii - 2, ii - 1, ii + 1, ii + 2]
                xx = [x for x in xx if x < series_cleaned.shape[0] and x >= 0]
                assert(len(xx) > 0)
                assert(not np.isnan(series_cleaned.iloc[xx]).to_numpy().all())
                series_cleaned.iloc[ii] = np.nanmedian(series_cleaned.iloc[xx].to_numpy().flatten())
    
    return series_cleaned, outliers


#def mape(predictions, actuals):
#    """Mean absolute percentage error"""
#    return ((predictions - actuals).abs() / actuals.abs()).mean()


def create_evaluation_df(predictions, test_inputs, H, scaler):
    """Create a data frame for easy evaluation"""
    eval_df = pd.DataFrame(predictions, columns=['t+'+str(t) for t in range(1, H+1)])
    eval_df['timestamp'] = test_inputs.dataframe.index
    eval_df = pd.melt(eval_df, id_vars='timestamp', value_name='prediction', var_name='h')
    eval_df['actual'] = np.transpose(test_inputs['target']).ravel()
    eval_df[['prediction', 'actual']] = scaler.inverse_transform(eval_df[['prediction', 'actual']])
    return eval_df


class TimeSeriesTensor(UserDict):
    """A dictionary of tensors for input into the RNN model.
    
    Use this class to:
      1. Shift the values of the time series to create a Pandas dataframe containing all the data
         for a single training example
      2. Discard any samples with missing values
      3. Transform this Pandas dataframe into a numpy array of shape 
         (samples, time steps, features) for input into Keras

    The class takes the following parameters:
       - **dataset**: original time series
       - **target** name of the target column
       - **H**: the forecast horizon
       - **tensor_structures**: a dictionary discribing the tensor structure of the form
             { 'tensor_name' : (range(max_backward_shift, max_forward_shift), [feature, feature, ...] ) }
             if features are non-sequential and should not be shifted, use the form
             { 'tensor_name' : (None, [feature, feature, ...])}
       - **freq**: time series frequency (default 'H' - hourly)
       - **drop_incomplete**: (Boolean) whether to drop incomplete samples (default True)
    """
    
    def __init__(self, dataset, target, H, tensor_structure, freq='H', drop_incomplete=True):
        self.dataset = dataset
        self.target = target
        self.tensor_structure = tensor_structure
        self.tensor_names = list(tensor_structure.keys())
        
        self.dataframe = self._shift_data(H, freq, drop_incomplete)
        self.data = self._df2tensors(self.dataframe)
    
    def _shift_data(self, H, freq, drop_incomplete):
        
        # Use the tensor_structures definitions to shift the features in the original dataset.
        # The result is a Pandas dataframe with multi-index columns in the hierarchy
        #     tensor - the name of the input tensor
        #     feature - the input feature to be shifted
        #     time step - the time step for the RNN in which the data is input. These labels
        #         are centred on time t. the forecast creation time
        df = self.dataset.copy()
        
        idx_tuples = []
        for t in range(1, H+1):
            df['t+'+str(t)] = df[self.target].shift(t*-1, freq=freq)
            idx_tuples.append(('target', 'y', 't+'+str(t)))

        for name, structure in self.tensor_structure.items():
            rng = structure[0]
            dataset_cols = structure[1]
            
            for col in dataset_cols:
            
            # do not shift non-sequential 'static' features
                if rng is None:
                    df['context_'+col] = df[col]
                    idx_tuples.append((name, col, 'static'))

                else:
                    for t in rng:
                        sign = '+' if t > 0 else ''
                        shift = str(t) if t != 0 else ''
                        period = 't'+sign+shift
                        shifted_col = name+'_'+col+'_'+period
                        df[shifted_col] = df[col].shift(t*-1, freq=freq)
                        #df.assign({shifted_col: df[col].shift(t*-1, freq=freq)})
                        #df.loc[:, shifted_col] = df.loc[:, col].shift(t*-1, freq=freq)
                        idx_tuples.append((name, col, period))
                
        df = df.drop(self.dataset.columns, axis=1)
        idx = pd.MultiIndex.from_tuples(idx_tuples, names=['tensor', 'feature', 'time step'])
        df.columns = idx

        if drop_incomplete:
            df = df.dropna(how='any')

        return df
    
    def _df2tensors(self, dataframe):
        
        # Transform the shifted Pandas dataframe into the multidimensional numpy arrays. These
        # arrays can be used to input into the keras model and can be accessed by tensor name.
        # For example, for a TimeSeriesTensor object named "model_inputs" and a tensor named
        # "target", the input tensor can be acccessed with model_inputs['target']
    
        inputs = {}
        y = dataframe['target']
        y = y.values
        inputs['target'] = y

        for name, structure in self.tensor_structure.items():
            rng = structure[0]
            cols = structure[1]
            tensor = dataframe[name][cols].values
            if rng is None:
                tensor = tensor.reshape(tensor.shape[0], len(cols))
            else:
                tensor = tensor.reshape(tensor.shape[0], len(cols), len(rng))
                tensor = np.transpose(tensor, axes=[0, 2, 1])
            inputs[name] = tensor

        return inputs
       
    def subset_data(self, new_dataframe):
        
        # Use this function to recreate the input tensors if the shifted dataframe
        # has been filtered.
        
        self.dataframe = new_dataframe
        self.data = self._df2tensors(self.dataframe)
