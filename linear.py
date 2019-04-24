import numpy as np 
from math import sqrt
import matplotlib.pyplot as plt 

  
def estimate_coef(x, y): 
    # number of observations/points 
    n = np.size(x) 
  
    # mean of x and y vector 
    m_x, m_y = np.mean(x), np.mean(y) 
  
    # calculating cross-deviation and deviation about x 
    SS_xy = np.sum(y*x) - n*m_y*m_x 
    SS_xx = np.sum(x*x) - n*m_x*m_x
   
  
    # calculating regression coefficients 
    b_1 = SS_xy / SS_xx 
    b_0 = m_y - b_1*m_x 
    return(b_0, b_1)
    
def r_value(X,Y):
    xy = [(x*y) for (x,y) in zip(X,Y)]
    x_sq = [(x**2) for x in X]
    y_sq = [(y**2) for y in Y]

    x_sum, y_sum, xy_sum, x_sq_sum, y_sq_sum = sum(X), sum(Y), sum(xy), sum(x_sq), sum(y_sq)
    r = (len(X) * xy_sum - x_sum * y_sum) / (sqrt( (len(X)*x_sq_sum - x_sum**2) * (len(Y)*y_sq_sum - y_sum**2) ))
    return r
  
def plot_regression_line(x, y, b): 
    # plotting the actual points as scatter plot 
    plt.scatter(x, y, color = "m", 
               marker = "o", s = 30) 
  
    # predicted response vector 
    y_pred = b[0] + b[1]*x 
  
    # plotting the regression line 
    plt.plot(x, y_pred, color = "g") 
  
    # putting labels 
    plt.xlabel('x') 
    plt.ylabel('y') 
  
    # function to show plot 
    plt.show() 
  
def main(): 
    # observations 
    x = np.array([10, 9, 2, 15, 10, 16, 11, 16]) 
    y = np.array([95, 80, 10, 50, 45, 98, 38, 93]) 
  
    # estimating coefficients 
    b = estimate_coef(x, y) 
    r = r_value(x,y)
    print("Estimated coefficients:\nb_0 = {} \nb_1 = {} ".format(b[0], b[1])) 
    
    print("Correlation measure is %8.2f" % r)
    print("R-square value is %8.2f" % (r * r * 100))
  
    # plotting regression line 
    plot_regression_line(x, y, b) 
  
if __name__ == "__main__": 
    main() 

