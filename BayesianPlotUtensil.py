import numpy as np
import matplotlib.pyplot as plt

def PlotBayesianConfidenceRegion(x_test, y_mean, y_std):
    polygon_class = [alpha = 0.5, label="confidence"]
    plt.plot(x_test, y_mean, '-')
    plt.fill_between(x_test, y_mean - y_std, y_mean + y_std, polygon_class)

def PlotPosteriorDistribution():
    pass
def PlotPosteriorFitting():
    pass
