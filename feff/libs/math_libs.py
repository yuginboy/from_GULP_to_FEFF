'''
* Created by Zhenia Syryanyy (Yevgen Syryanyy)
* e-mail: yuginboy@gmail.com
* License: this code is in GPL license
* Last modified: 2017-02-09
'''
import numpy as np
def approx_jacobian(x,func,epsilon=1e-3,*args):
    """Approximate the Jacobian matrix of callable function func

       * Parameters
         x       - The state vector at which the Jacobian matrix is
desired
         func    - A vector-valued function of the form f(x,*args)
         epsilon - The peturbation used to determine the partial derivatives
         *args   - Additional arguments passed to func

       * Returns
         An array of dimensions (lenf, lenx) where lenf is the length
         of the outputs of func, and lenx is the number of

       * Notes
         The approximation is done using forward differences
         from: https://mail.scipy.org/pipermail/scipy-user/2008-November/018725.html

    """
    x0 = np.asfarray(x)
    f0 = func(*((x0,)+args))
    jac = np.zeros([np.size(x0), np.size(f0)])
    dx = np.zeros(len(x0))
    for i in range(len(x0)):
       dx[i] = epsilon
       jac[i] = (func(*((x0+dx,)+args)) - f0)/epsilon
       dx[i] = 0.0
    return np.array(jac.transpose())
import numpy as np

def approx_hessian1d(x, func, epsilon=1e-3, emin=1e-3, *args):
    """
    Calculate the hessian matrix with finite differences
    Parameters:
       * Parameters
         x       - The state vector at which the Hessian matrix is desired
         func    - A one function of the form f(x,*args)
         epsilon - The vector of peturbation used to determine the partial derivatives
         emin - minimum value of epsilon
         *args   - Additional arguments passed to func
    Returns:
       an array of shape (x.dim, x.ndim) + x.shape
       where the array[i, j, ...] corresponds to the second derivative x_ij
    """
    x0 = np.asfarray(x)
    f0 = func(*((x0,) + args))
    hes = np.zeros([np.size(x0), np.size(x0)])
    if np.size(epsilon) < 2:
        epsilon = x0*0.0 + epsilon
    # check epsilon vector for nonzeros values:
    for i, eps in enumerate(epsilon):
        if abs(eps) < emin:
            epsilon[i] = emin

    dy = np.zeros(len(x0))
    dx = np.zeros(len(x0))
    for i in range(len(x0)):
        dx[i] = epsilon[i]
        for j in range(len(x0)):
            dy[j] = epsilon[j]
            hes[i, j] = ( func(*((x0 + dx + dy,) + args)) - func(*((x0 + dx,) + args))
                         - func(*((x0 + dy,) + args)) + f0 ) / (epsilon[i]*epsilon[j])
            dy[j] = 0.0
        dx[i] = 0.0
    return hes

def approx_hessian1d_diag(x, func, epsilon=1e-3, emin=1e-12, *args):
    """
    defines the order of the error term in the Taylor approximation used.
    Calculate the hessian matrix with finite differences
    Parameters:
       * Parameters
         x       - The state vector at which the Hessian matrix is desired
         func    - A one function of the form f(x,*args)
         epsilon - The vector of peturbation used to determine the partial derivatives
         emin - minimum value of epsilon
         *args   - Additional arguments passed to func
    Returns:
       an array of shape (x.dim, x.ndim) + x.shape
       where the array[i, j, ...] corresponds to the second derivative x_ij
    """
    x0 = np.asfarray(x)
    f0 = func(*((x0,) + args))
    hes = np.zeros([np.size(x0), np.size(x0)])
    if np.size(epsilon) < 2:
        epsilon = x0*0.0 + epsilon
    # check epsilon vector for nonzeros values:
    for i, eps in enumerate(epsilon):
        if abs(eps) < emin:
            epsilon[i] = emin

    dxy = np.zeros(len(x0))
    dy = np.zeros(len(x0))
    dx = np.zeros(len(x0))
    for i in range(len(x0)):
        dx[i] = epsilon[i]
        hes[i, i] = ( func(*((x0 + 2*dx,) + args)) - 2*func(*((x0 + dx,) + args)) + f0 ) / (epsilon[i]**2)
        dx[i] = 0.0
    return hes

def approx_errors(func, x0, epsilon=1e-8):
    '''

    :param fun: function which we will use for calculation
    :param x0: The state vector at which errors are desired
    :param epsilon: vector of dx values. For problems when dx couldn't be less then certain value
    :return: vector of standard error
    I think errors will be: sqrt(sigma_squared)*se
    '''
    #   Yudi Pawitan writes in his book In All Likelihood that the second derivative of the log-likelihood evaluated at
    # the maximum likelihood estimates (MLE) is the observed Fisher information (see also this document, page 2).
    # This is exactly what most optimization algorithms like optim in R return: the Hessian evaluated at the MLE.
    hess = approx_hessian1d_diag(x0, func, epsilon=epsilon)
    se = np.zeros(np.size(x0))
    se = 1/np.sqrt(abs(np.diag(hess)))
    # change inf number
    se[np.isinf(se)] = 0
    se[np.isneginf(se)] = 0
    return se

def sigma_squared (X, Y, func, p0):
    '''

    :param X: x data
    :param Y: y data
    :param func: theoretical function of estimation y-data from the x-data
    :param p0: parameters for func (ex: coefficients)
    :return: unbiased sample variance
    '''
    sum = 0
    Y2 = func(X, p0)
    if ( len(X) == len (Y) ) :
        if ( len(Y) == len (Y2) ):
            sum = np.sum(  (Y2 - Y)**2  ) / (len(Y)-len(p0))
        else:
            print ('==> sigma_squared error: len Y-data points and func(X, p0)-data points is not equal')
    else:
        print ('==> sigma_squared error: len X-data points and Y-data points is not equal')
    return sum

