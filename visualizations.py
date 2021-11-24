import matplotlib.pyplot as plt

def plot_bar(data,factor):
    """
    Plots bucketed returns
    """
    
    plot = data[[factor,'target-1','target-2','target-3','target-4','target-5']].groupby(factor).mean().plot.bar()
    plt.title('Factor of'+ ' '+ factor)    
    
    return plot