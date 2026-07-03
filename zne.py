import numpy as np

from scipy.optimize import curve_fit
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel, WhiteKernel

from qiskit import QuantumCircuit

def fold_circuit(qc, lambda_scale, strategy="global"):
    if lambda_scale % 2 == 0:
        raise ValueError("Scale factor must be an odd integer.")
        
    n_folds = (lambda_scale - 1) // 2
    qc_folded = QuantumCircuit(qc.num_qubits)
    
    if strategy == "global":
        for _ in range(n_folds):
            qc_folded.compose(qc.inverse(), inplace=True)
            qc_folded.compose(qc, inplace=True)
    elif strategy == "local":
        for instruction in qc.data:
            if len(instruction.qubits) > 1:
                for _ in range(n_folds):
                    qc_folded.append(instruction.operation.inverse(), instruction.qubits)
                    qc_folded.append(instruction)
                    
    return qc_folded

def extrapolate_zne(lambda_scales, expvals, method="gaussian_process", prior_mean=0.0, prior_var=1.0):
    x = np.array(lambda_scales)
    y = np.array(expvals)
    
    if method == "linear":
        # mx + b
        coeffs = np.polyfit(x, y, 1)
        
        return coeffs[-1]
    elif method == "bayesian_linear":
        X = np.vstack([np.ones(len(x)), x]).T
        Y = np.array(y)
        
        alpha = 1.0 / prior_var
        I = np.eye(2)
        I[0, 0] = 0
        
        beta = np.linalg.inv(X.T @ X + alpha * I) @ X.T @ Y
        
        return beta[0]
    elif method == "richardson":
        # Polynomial
        degree = len(x) - 1
        coeffs = np.polyfit(x, y, degree)
        
        return coeffs[-1]
    elif method == "exponential":
        #  A * e^(-B * x)
        def exp_func(l, a, b):
            return a * np.exp(-b * l)
        
        guess = [y[0], 0.1]
        
        try:
            popt, _ = curve_fit(exp_func, x, y, p0=guess, maxfev=2000)
            return popt[0]
        except RuntimeError:
            print("Exponential fit failed to converge. Falling back to linear.")
            
            return np.polyfit(x, y, 1)[-1]
    elif method == "gaussian_process":
        # Gaussin(Constant (Amplitude) * RBF (Smoothness) + WhiteKernel (Shot Noise))
        kernel = ConstantKernel(1.0, (1e-3, 1e3)) * RBF(length_scale=2.0, length_scale_bounds=(0.5, 10.0)) \
                 + WhiteKernel(noise_level=1e-4, noise_level_bounds=(1e-6, 1e-2))
                 
        gp = GaussianProcessRegressor(kernel=kernel, n_restarts_optimizer=10, normalize_y=False)
        
        X_train = x.reshape(-1, 1)
        gp.fit(X_train, y)
        
        y_pred, sigma = gp.predict([[0.0]], return_std=True)
        
        return y_pred[0]
    else:
        raise ValueError(f"Unknown extrapolation method: {method}")
