import pickle
import pandas as pd

def plot_result(filename):

    with open(filename,'rb') as f:
        data = pickle.load(f)

    df = pd.DataFrame({'data':data['scores']})
    df['rolling_mean'] = df['data'].rolling(100).mean()
    df.plot()
import matplotlib.pyplot as plt  
def plot_result_grid(filenames,
                     n_rows,
                     n_columns,
                     fig_size = None
                     ):

    fig = plt.figure(figsize= fig_size or (7,7))
    fig.subplots_adjust(hspace=0.3)
    for i,filename in enumerate(filenames):
        ax = fig.add_subplot(n_rows,n_columns, i+1)
        with open(filename,'rb') as f:
            data = pickle.load(f)

        df = pd.DataFrame({'data':data['scores']})
        df['rolling_mean'] = df['data'].rolling(100).mean()
        # df['lb'] = df['rolling_mean']-df['data'].rolling(100).std()
        # df['ub'] = df['rolling_mean']+df['data'].rolling(100).std()
        df.plot(ax =ax)
        ax.set_title(data['hyperparams']['description'])
    plt.show()
