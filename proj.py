import numpy as np
from sklearn.linear_model import LinearRegression, LogisticRegression
from utils import est_Cov, is_one_hot, est_Cov_torch
import sys
import gc

# for the optimization
from pymanopt.manifolds import Stiefel
from pymanopt.optimizers import TrustRegions, ConjugateGradient, SteepestDescent
from pymanopt import Problem
import pymanopt
import autograd.numpy as anp
from pymanopt.function import autograd

# for the pytorch optimization
import torch
import mctorch.nn as mnn
import mctorch.optim as moptim
from torch.utils.data import DataLoader, TensorDataset

class proj:
    """    
    This class computes projection matrices that remove protected information (z) from data (X).
    In some cases, this also preserves information about a target variable (y).
    
    Attributes:
        eps (float): Small constant for numerical stability (default: 1e-8)
    """
    
    def __init__(self):
        # Numerical stability threshold
        self.eps = 1e-6
        self.causal_LEACE_variant = None
        
    def stable_P_computation(self, V, U, I_d, whiten=False, W=None, W_inv=None):
    
        # 1. More stable oblique projection computation
        VTU = V.transpose(1, 0) @ U
        
        # Check if VTU is invertible
        det_val = torch.det(VTU @ VTU.transpose(1, 0))
        if torch.abs(det_val) < torch.tensor(1e-10, dtype=torch.float32):
            # Use pseudoinverse for singular case
            VT_inv_partial = torch.linalg.pinv(VTU) @ V.transpose(1, 0)
        else:
            # Use more stable computation - avoid explicit inverse when possible
            # For overdetermined case, use least squares solution
            if VTU.shape[0] >= VTU.shape[1]:
                VTU_T_VTU = VTU.transpose(1, 0) @ VTU
                VT_inv_partial = torch.linalg.solve(VTU_T_VTU , 
                                                VTU.transpose(1, 0) @ V.transpose(1, 0))
            else:
                VTU_VTU_T = VTU @ VTU.transpose(1, 0)
                eps_eye_overdetermined = self.eps * torch.eye(VTU.shape[0], dtype=torch.float32)
                VT_inv_partial = torch.linalg.solve(VTU_VTU_T + eps_eye_overdetermined, 
                                                VTU @ V.transpose(1, 0))
        
        if whiten and W is not None and W_inv is not None:
            # If whitening is applied, adjust the projection matrix accordingly
            P = I_d - W_inv @ (U @ VT_inv_partial) @ W  # Oblique projection matrix with whitening
        else:
            P = I_d - U @ VT_inv_partial # Oblique projection matrix
        
        
        return P
        
    def opt_U_torch(self, X, z, y, Theta=None, w=None, lr=1e-2, tol=1e-3, max_epochs=100, verbose=True, patience=10, initial='SPLICE'):
        """
        Optimize range of projection using PyTorch with epoch-based training and early stopping.
        
        Args:
            X (np.ndarray): Data matrix of shape (n, d)
            z (np.ndarray): Protected attribute, shape (n,) or (n, k_z) 
            y (np.ndarray): Target attribute, shape (n,) or (n, k_y)
            Theta (np.ndarray): Logistic regression coefficients, shape (d, k_y)
            w (np.ndarray): Sample weights, shape (n, 1). If None, uses uniform weights
            lr (float): Learning rate
            tol (float): Tolerance for early stopping - halt if improvement < tol
            max_epochs (int): Maximum number of epochs
            batch_size (int): Batch size for training. If None, uses full batch
            verbose (bool): Print progress
            patience (int): Number of epochs to wait for improvement before stopping
            
        Returns:
            tuple: (P_best, training_history) where P_best is the best projection matrix
                and training_history is a dict with loss and accuracy history
        """
        
        # Handle dimensions - reshape to (n, k) format
        n = X.shape[0]
        z = z.reshape(n, -1) if z.ndim == 1 else z
        y = y.reshape(n, -1) if y.ndim == 1 else y
        
        # Convert to torch tensors
        X_torch = torch.tensor(X, dtype=torch.float32)
        z_torch = torch.tensor(z, dtype=torch.float32)
        y_torch = torch.tensor(y, dtype=torch.float32)
        w_torch = torch.tensor(np.ones((n, 1)) if w is None else w.reshape(-1, 1), dtype=torch.float32)
        
        # Handle Theta - fit for multi-output if needed
        if Theta is None:
            if y.shape[1] == 1:
                # Binary classification
                Theta = LogisticRegression(fit_intercept=False).fit(X, y.ravel()).coef_.T
            else:
                # Multi-output classification - fit separate models
                Theta = []
                for i in range(y.shape[1]):
                    logreg = LogisticRegression(fit_intercept=False)
                    logreg.fit(X, y[:, i])
                    Theta.append(logreg.coef_.ravel())
                Theta = np.column_stack(Theta)
        Theta_torch = torch.tensor(Theta, dtype=torch.float32)
        
        # Calculate dimensions and basis
        d = X.shape[1]
        
        # For multi-dimensional z, calculate covariance for each column and concatenate
        if z.shape[1] == 1:
            Cov_xz = est_Cov(X, z.ravel()).reshape(d, 1)
        else:
            Cov_xz = []
            for i in range(z.shape[1]):
                cov_i = est_Cov(X, z[:, i]).reshape(d, 1)
                Cov_xz.append(cov_i)
            Cov_xz = np.hstack(Cov_xz)
        
        # Calculate rank and orthonormal basis
        c = np.linalg.matrix_rank(Cov_xz)
        if c == 0:
            # Handle case where there's no correlation
            raise ValueError("Covariance matrix has zero rank. Cannot compute projection.")
            
        # Pre-compute constants
        I_d = torch.eye(d, dtype=torch.float32)
        eps = self.eps
        U = self.get_basis(Cov_xz)
        
        # Calculate weighted covariance matrix
        Sigma = X.T @ X
        Sigma_inv = np.linalg.pinv(Sigma)
        if initial == 'LEACE':
            
            # Initialize manifold parameter
            V = torch.tensor(Sigma_inv @ U, dtype=torch.float32, requires_grad=True)
            W, W_inv = None, None
            
        elif initial == 'random':
            # Random initialization on Stiefel manifold
            V = torch.randn(d, c, dtype=torch.float32, requires_grad=True)
            V, _ = torch.linalg.qr(V)
            W, W_inv = None, None

        elif initial == 'SPLICE':
            
            # get the whitening matrix (as torch.tensor)
            W = torch.tensor(self.est_W(X), dtype=torch.float32)
            W_inv = torch.linalg.pinv(W)
            
            # get cov(x, z) and cov(x, y)
            Cov_Wxz = est_Cov_torch( X_torch @ W, z_torch)
            Cov_Wxy = est_Cov_torch( X_torch @ W, y_torch)
            
            # Initialize using SPLICE method
            W_Cov = np.hstack((Cov_Wxz, Cov_Wxy))
            U_W_Cov = self.get_basis(W_Cov)
            u_W_xy = self.get_basis(Cov_Wxy)
            I = torch.eye(d, dtype=torch.float32)
            u_W_xy_perp =  (I - u_W_xy  @ u_W_xy.T) @ U_W_Cov
            V = self.get_basis(u_W_xy_perp)
            
            # get basis U
            U = self.get_basis(Cov_Wxz)
            
            # turn V, U to torch tensors
            V = torch.tensor(V, dtype=torch.float32)
            
        # fill in the V_W, U_W
        whiten = True if initial == 'SPLICE' else False
        U_torch = torch.tensor(U, dtype=torch.float32)
        P_initial = self.stable_P_computation(V, U_torch, I_d, whiten=whiten, W=W, W_inv=W_inv)

        # Setup optimizer
        optimizer = torch.optim.LBFGS([V], lr=lr, max_iter=1000, line_search_fn='strong_wolfe',)
        # Ensure V is a torch tensor with requires_grad=True
        V.requires_grad = True

        # Training history
        history = {
            "loss": [],
            "accuracy": [],
            "epoch": []
        }
        
        # Early stopping variables
        best_loss = float('inf')
        best_V = V.clone().detach()
        epochs_without_improvement = 0
        
        # Initial evaluation
        with torch.no_grad():
            initial_loss = self.compute_loss_torch(V, X_torch, y_torch, w_torch, U_torch, I_d, Theta_torch, eps, 
                                                   whiten=whiten, W=W, W_inv=W_inv)
            y_pred = torch.round(torch.sigmoid(X_torch @ P_initial.transpose(1, 0) @ Theta_torch))
            initial_acc = torch.mean((y_pred == y_torch).float())
            
            history["loss"].append(initial_loss.item())
            history["accuracy"].append(initial_acc.item())
            history["epoch"].append(0)
            
            if verbose:
                print(f"Initial - Loss: {initial_loss:.6f}, Accuracy: {initial_acc:.4f}")
                print(f"V.shape: ({d}, {c}), z.shape: {z.shape}, y.shape: {y.shape}")
                        
                
        # Define closure for LBFGS optimizer
        def closure():
            optimizer.zero_grad()
            loss = self.compute_loss_torch(V, X_torch, y_torch, w_torch, U_torch, I_d, Theta_torch, eps,
                                           whiten=whiten, W=W, W_inv=W_inv)
            loss.backward()
            return loss
                
        # Training loop
        for epoch in range(1, max_epochs + 1):
            # Reset epoch variables
            epoch_loss = 0.0
            
           
            # Full batch training
            optimizer.step(closure)
            epoch_loss = self.compute_loss_torch(V, X_torch, y_torch, w_torch, U_torch, I_d, Theta_torch, eps, 
                                                   whiten=whiten, W=W, W_inv=W_inv)
            
            # Evaluate on full dataset
            with torch.no_grad():
                P_current = self.stable_P_computation(V, U_torch, I_d, whiten=whiten, W=W, W_inv=W_inv)
                y_pred = torch.round(torch.sigmoid(X_torch @ P_current.transpose(1, 0) @ Theta_torch))
                
                # calculate the weighted accuracy using w_torch
                correct_predictions = (y_pred == y_torch).float()
                current_acc = torch.mean(correct_predictions * w_torch)
                
                history["loss"].append(epoch_loss)
                history["accuracy"].append(current_acc.item())
                history["epoch"].append(epoch)
            
            # Check for improvement
            improvement = best_loss - epoch_loss
            if improvement > tol:
                best_loss = epoch_loss
                best_V = V.clone().detach()
                epochs_without_improvement = 0
            else:
                epochs_without_improvement += 1
            
            # Verbose output
            if verbose and (epoch % 1 == 0):
                print(f"Epoch {epoch:4d} - Loss: {epoch_loss:.6f}, Accuracy: {current_acc:.4f}, "
                    f"Improvement: {improvement:.6f}, epochs without improvement: {epochs_without_improvement}")
            
            # Early stopping check
            if epochs_without_improvement >= patience:
                if verbose:
                    print(f"Early stopping at epoch {epoch}. No improvement for {patience} epochs.")
                break
        
        # Calculate final projection matrix using best V
        with torch.no_grad():
            P_best = self.stable_P_computation(best_V, U_torch, I_d, whiten=whiten, W=W, W_inv=W_inv)
            y_pred_best = torch.round(torch.sigmoid(X_torch @ P_best.transpose(1, 0) @ Theta_torch))
            best_accuracy = torch.mean((y_pred_best == y_torch).float())
            
            if verbose:
                print(f"Optimization completed. Best loss: {best_loss:.6f}, Best accuracy: {best_accuracy:.4f}")
        
        return P_best.detach().cpu().numpy()

    def compute_loss_torch(self, V, X, y, w_batch, U, I_d, Theta, eps, whiten=False, W=None, W_inv=None):
        """
        Compute loss for a batch of data.
        
        Args:
            V: Current parameter tensor
            X, y, w_batch: Batch tensors
            U, I_d, Theta, eps: Pre-computed constants
            
        Returns:
            torch.Tensor: Scalar loss value
        """
        # Compute projection matrix
        P = self.stable_P_computation(V, U, I_d, whiten=whiten, W=W, W_inv=W_inv)
        
        # Efficient prediction computation
        P_Theta = P.transpose(1, 0) @ Theta
        y_logits = X @ P_Theta
        
      
        # Stable sigmoid computation
        y_prob = torch.where(y_logits >= 0,
                            1 / (1 + torch.exp(-y_logits)),
                            torch.exp(y_logits) / (1 + torch.exp(y_logits)))
        
        y_prob_clipped = torch.clamp(y_prob, eps, 1 - eps)
        
        # Binary cross-entropy loss
        log_prob = torch.log(y_prob_clipped)
        log_one_minus_prob = torch.log(1 - y_prob_clipped)
        
        if y.shape[1] == 1:
            # Single output
            loss_per_sample = -(y * log_prob + (1 - y) * log_one_minus_prob)
        else:
            # Multi-output - sum over outputs
            loss_per_sample = -torch.sum(y * log_prob + (1 - y) * log_one_minus_prob, dim=1, keepdim=True)
        
        # Apply weights
        weighted_loss = w_batch * loss_per_sample
        return torch.mean(weighted_loss)

            
    def get_basis(self, v):
        """
        Compute orthonormal basis for the span of the given matrix v.
        
        Args:
            v (np.ndarray): Matrix of shape (d, k)
            
        Returns:
            np.ndarray: Matrix of shape (d, rank(v)) whose columns form orthonormal basis
                    for the span of v
            
        Note:
            Handles the zero matrix case by returning empty basis
        """
        d, k = v.shape
        
        # Handle zero matrix case
        if np.allclose(v, 0):
            return np.zeros((d, 0))  # Empty basis for zero space
        
        # Perform QR decomposition on v
        Q, R = np.linalg.qr(v)
        
        # Find the rank by counting non-zero diagonal elements of R
        rank = np.sum(np.abs(np.diag(R)) > 1e-10)
        
        # Return the first 'rank' columns of Q (basis for span of v)
        basis = Q[:, :rank]
        
        return basis
        
    

    def fit(self, X, z, y, method='LEACE', info_type='Cov', coef=None, w=None):
        """
        Determine the P and b for the projection matrix.

        Args:
            X (np.ndarray): Data matrix of shape (n, d)
            z (np.ndarray): Protected attribute to remove, shape (n, 1)
            y (np.ndarray): Target information to preserve, shape (n, k)
            method (str): Method to use for projection (default: 'LEACE')
        
        """
        # demean X
        self.x_mean = np.mean(X, axis=0)
        X_demeaned = X - self.x_mean

        if method == 'LEACE':
            
            # get the projection matrix and bias term from LEACE
            self.P = self.LEACE(X_demeaned, z)
            self.b = (self.x_mean - self.P @ self.x_mean)

        elif method == 'SPLINCE':
            # get the projection matrix from optimal separation projection
            self.P = self.opt_sep_proj(X_demeaned, z, y, info_type=info_type, coef=coef)
            self.b = (self.x_mean - self.P @ self.x_mean)

        elif method == 'LEACE-no-whitening':

            # get P based on orthogonal projection
            cov_x_z = est_Cov(X_demeaned, z)
            v = cov_x_z / np.linalg.norm(cov_x_z)
            self.P = self.get_orth_proj(v)

            # get the bias term
            self.b = (self.x_mean - self.P @ self.x_mean)
        
        elif method == 'causal-LEACE':

            include_E_X_y_range = True if self.causal_LEACE_variant == 'range' else False
            self.P, self.reg_X_y, self.x_mean_tilde = self.causal_LEACE(X_demeaned, z, y, include_E_X_y_range=include_E_X_y_range)
            self.b = (self.x_mean - self.P @ self.x_mean)
        
        elif method == 'SAL':

            # get the projection matrix and bias term from SAL
            self.P = self.SAL(X_demeaned, z)
            self.b = np.zeros(X.shape[1])
            
        elif method == 'optimized-range':
            
            # get the projection matrix from optimized range
            self.P = self.opt_U_torch(X_demeaned, z, y, w=w, Theta=coef)
            self.b = (self.x_mean - self.P @ self.x_mean)
        
        self.method = method

    def set_causal_LEACE_variant(self, variant):

        self.causal_LEACE_variant = variant
        
    def apply_projection(self, X, y=None):
        """
        Apply the projection matrix to the data.
        """

        # if method=='causal-LEACE', then we adapt the application based on the variant
        if self.method == 'causal-LEACE':

            # variant where we use knowledge of y - either based on an oracle or prediction
            if self.causal_LEACE_variant == 'use_y':
                if y is None:
                    sys.exit("Need to provide y for causal LEACE")
                else:

                    # we require here that y is one-hot encoded (even if binary)
                    if y.shape[1] == 1:
                         y = np.concatenate([1 - y, y], axis=1)

                    # get the residuals
                    X_cond_y = self.reg_X_y.predict(y)
                    X_tilde = X - X_cond_y
                    
                    # apply the projection
                    X_proj = X_tilde @ self.P.T + X_cond_y 

            elif self.causal_LEACE_variant == 'range':
                X_proj =  X @ self.P.T + self.b
            
        else:
            X_proj =  X @ self.P.T + self.b

        return X_proj

    def SAL(self, X, z):
        """
        Implement the spectral attribute removal (SAL) method.

        Args:
            X (np.ndarray): Data matrix of shape (n, d)
            z (np.ndarray): Protected attribute to remove, shape (n, k)

        Returns:
            np.ndarray: Projection matrix P of shape (d, d)
        """

        # Calculate the cross-covariance matrix
        Sigma = est_Cov(X, z)

        # perform SVD on Sigma
        U, _, _ = np.linalg.svd(Sigma)

        # Use the rank of Sigma to determine the number of components to keep
        k = np.linalg.matrix_rank(Sigma)

        # Get the first rank columns of U
        U_k = U[:, :k]
        
        # Get the orthogonal complement of U_k
        I_d = np.eye(X.shape[1])
        P = I_d - np.matmul(U_k, U_k.T)

        return P


    def LEACE(self, X, z):  
        """
        Determine the projection matrix using LEACE.

        Args:
            X (np.ndarray): Data matrix of shape (n, d)
            z (np.ndarray): Protected attribute to remove, shape (n, k)
        
        Returns:
            np.ndarray: Projection matrix P of shape (d, d)
        """

        # first, get the Whitening matrix
        W = self.est_W(X)

        # whiten the data
        XW = X @ W

        # check: if z.ndim > 1 and it is one-hot encoded, we can remove the last column
        if z.ndim > 1 and is_one_hot(z):
            z = z[:, :-1]
            print('Removing last column of z, as it is one-hot encoded')

        # second, get the cross-covariance with z and X
        v = est_Cov(X @ W, z)
        
        # calculate the pinv
        if v.ndim > 1:
            v_pinv = np.linalg.pinv(v)
        else:
            norm_sq = np.linalg.norm(v)**2
            v_pinv = (v / norm_sq).T

        # define the orthogonal projection matrix
        P_proj_v = v @ v_pinv
        
        # using this, define the projection matrix
        W_pinv = np.linalg.pinv(W)
        I_d = np.eye(W_pinv.shape[0])
        P = I_d - W_pinv @ P_proj_v @ W

        # delete the intermediate variables
        del W, XW, v,  v_pinv, P_proj_v
        gc.collect()

        return P

    
    def causal_LEACE(self, X, z, y, include_E_X_y_range=False):
        """
        
        Args:
            X (np.ndarray): Data matrix of shape (n, d)
            z (np.ndarray): Protected attribute to remove, shape (n, 1)
            y (np.ndarray): Target information to preserve, shape (n, k)
        
        Returns:
            np.ndarray: Projection matrix P of shape (d, d)
        """

        # we require here that y is one-hot encoded (even if binary)
        if len(y.shape) == 1:
            y = y.reshape(-1, 1)
        if y.shape[1] == 1:
            y = np.concatenate([1 - y, y], axis=1)

        # first, we calculate the conditional mean of X given y for each value of y
        reg_X_y = LinearRegression().fit(y, X)
        X_cond_y = reg_X_y.predict(y)

        # second, calculate the conditional mean of y given z
        reg_y_z = LinearRegression().fit(y, z)
        z_cond_y = reg_y_z.predict(y)

        # now calculate the residuals
        X_tilde = X - X_cond_y
        z_tilde = z - z_cond_y

        # apply LEACE to X_tilde and z_tilde
        if include_E_X_y_range:
            E_X_y = reg_X_y.coef_
            Cov_X_tilde_z = est_Cov(X_tilde, z_tilde)
            v = Cov_X_tilde_z / np.linalg.norm(Cov_X_tilde_z)
            P = self.get_oblique_proj(v, E_X_y)
        else:
            P = self.LEACE(X_tilde, z_tilde)
       
        # get the mean of x_tilde (per y)
        y_unique = np.unique(y, axis=0)
        x_mean_tilde = np.zeros((y_unique.shape[0], X.shape[1]))
        i=0
        for y_val in y_unique:
            y_val_index = np.where((y == y_val).all(axis=1))[0]
            X_tilde_y = X_tilde[y_val_index]
            x_mean_tilde[i, :] = np.mean(X_tilde_y, axis=0)
            i += 1
           

        return P, reg_X_y, x_mean_tilde

    def est_W(self, X):
        """
        Estimate the whitening matrix for data matrix X.
        
        Args:
            X (np.ndarray): Data matrix of shape (n, d)
            
        Returns:
            np.ndarray: Whitening matrix of shape (d, d)
        
        Note:
            Uses eigendecomposition and handles numerical stability through thresholding
        """

        # get the covariance matrix
        Cov_X = np.cov(X, rowvar=False)

        # get the eigenvalues and eigenvectors
        eigenvals, eigenvecs = np.linalg.eigh(Cov_X)

        # for numerical stability
        eigenvals[eigenvals < self.eps] = 0

        # Take reciprocal square root of non-zero eigenvalues
        diag = np.zeros_like(eigenvals)
        nonzero = eigenvals > 0
        diag[nonzero] = 1.0 / np.sqrt(eigenvals[nonzero])
        
        # Compute whitening matrix
        W = np.dot(eigenvecs, np.dot(np.diag(diag), eigenvecs.T))

        return W
    
    def find_orthogonal_vector(self, u, v):
        """
        Find a vector orthogonal to u in the span of u and v.
        
        Given two linearly independent vectors u and v, finds a vector w that is:
        1. In the span of u and v
        2. Orthogonal to u
        3. Unit length
        
        Args:
            u (np.ndarray): First vector of shape (d, 1)
            v (np.ndarray): Second vector of shape (d, 1)
            
        Returns:
            np.ndarray: Orthogonal unit vector of shape (d, 1)
            
        Raises:
            ValueError: If either input vector is zero
            
        Note:
            Uses the construction w = au + bv where a,b are chosen to ensure w⋅u = 0
        """
        if np.allclose(u, 0) or np.allclose(v, 0):
            raise ValueError("Input vectors cannot be zero vectors")
            
        # Want w·u = 0 and w = au + bv
        # This means: (au + bv)·u = 0
        # So: a(u·u) + b(v·u) = 0
        # Let a = -(v·u), b = (u·u)
        a = -np.matmul(v.T, u)
        b = np.matmul(u.T, u)
        
        # Compute w
        w = a * u + b * v
        
        # Normalize
        w = w / np.linalg.norm(w)
        
        return w
    
    def get_orthogonal_complement_basis(self, v):
        """
        Compute orthonormal basis for the orthogonal complement of a vector.
        
        Uses QR decomposition to find d-1 vectors that form an orthonormal basis for
        the space orthogonal to v.
        
        Args:
            v (np.ndarray): Vector of shape (d, k)
            
        Returns:
            np.ndarray: Matrix of shape (d, d-k) whose columns form orthonormal basis
            
        Note:
            Handles the zero vector case by returning standard basis vectors
        """
        d = v.shape[0]
        k = v.shape[1] if v.ndim > 1 else 1

        # resphape if necessary
        if len(v.shape) == 1:
            v = v.reshape(-1, 1)
        
        # Normalize v
        u = v / np.linalg.norm(v, axis=0, keepdims=True)
        
        # Create a matrix with u as the first column and the rest as standard basis vectors
        A = np.eye(d)
        A[:, :k] = u
        
        # Perform QR decomposition
        Q, R = np.linalg.qr(A)
        
        # Return the last d-1 columns of Q, which form the orthogonal complement basis
        return Q[:, k:]
    
    def get_column_space_basis(self, A):
        """
        Get orthonormal basis for column space of matrix A using SVD.
        
        Args:
            A: Matrix of shape (d, d-1)
        Returns:
            Orthonormal basis vectors as columns of matrix
        """
        # Perform SVD
        U, s, Vh = np.linalg.svd(A, full_matrices=False)

        # Threshold for numerical stability
        tol = max(A.shape) * np.spacing(max(s))
        r = sum(s > tol)
        return U[:, :r]
    
    def opt_sep_proj(self, X, z, y, info_type='Cov', coef=None):
        """
        Compute optimal separation projection matrix.
        
        Finds a projection matrix P that:
        1. Projects parallel to the direction of z (removes this information)
        2. Preserves Covariance between X and y
        3. Maintains as much other information as possible
        
        Args:
            X (np.ndarray): Data matrix of shape (n, d)
            z (np.ndarray): Protected attribute to remove, shape (n, 1)
            y (np.ndarray): Target information to preserve, shape (n, 1)
            info_type (str): Type of information to preserve (default: 'Cov')
            coef (np.ndarray): Coefficients for linear regression (optional) of shape (d, k)
            
        Returns:
            np.ndarray: Projection matrix P of shape (d, d)
            
        Note:
            Uses whitening transformation for numerical stability
            Choice of u_type affects what information about y is preserved
        """

        # first, get the Whitening matrix
        W = self.est_W(X)

        # whiten the data
        XW = np.matmul(X, W)
        
        # check: if z.ndim > 1 and it is one-hot encoded, we can remove the last column
        if z.ndim > 1 and is_one_hot(z):
            z = z[:, :-1]
            print('Removing last column of z, as it is one-hot encoded')

        # second, get the cross-covariance with z and X
        v = est_Cov(XW, z)

            
        
        # third, use covariance to estimate the vector to be preserved
        if info_type == 'Cov':
            # get the covariance with y
            u_k = est_Cov(XW, y)
        elif info_type == 'coef':
            if coef is None:
                raise ValueError("Need to provide coef for info_type='coef'")
            u_k = np.matmul(W, coef)
        else:
            raise ValueError("info_type must be either 'Cov' or 'coef'")

        # then, get the projection matrix
        P_prime = self.get_oblique_proj(v, u_k)
        self.P_prime = P_prime
       
        # using this, define the projection matrix
        W_pinv = np.linalg.pinv(W)
        P_star = W_pinv @ P_prime @ W

        return P_star
    


    def get_orth_proj(self, v):
        """
        Compute orthogonal projection matrix projecting parallel to v.
        
        Creates a projection matrix P that:
        1. Projects orthogonally onto the space perpendicular to v
        2. Satisfies P² = P (projection property)
        3. Is symmetric
        
        Args:
            v (np.ndarray): Direction vector of shape (d, 1)
            
        Returns:
            np.ndarray: Projection matrix of shape (d, d)
        """

        # check if the vector is unit
        if np.abs(np.linalg.norm(v) - 1) > self.eps:
            v = v/np.linalg.norm(v)

        # get the dimension of the vector
        d = v.shape[0]

        # get d x (d-1) matrix whose columns are orthonormal basis of the set of all vectors orthogonal to v
        I_d = np.eye(d)
        
        # get the projection matrix
        P = I_d - np.matmul(v, v.T)

        return P


    # Function to orthogonalize a candidate vector against the current basis
    def orthogonalize(self, vec, basis_list):
        for b in basis_list:
            vec = vec - np.dot(b, vec) * b
        return vec
        



    def get_oblique_proj(self, v, u_k):
        """
        Compute oblique projection matrix with specific properties.
        
        Creates a projection matrix P that:
        1. Projects parallel to v
        2. Ensures each column vector in u_k is in the range of the projection matrix
        3. Satisfies P² = P (projection property)
        
        Args:
            v (np.ndarray): Direction to project parallel to, shape (d, k)
            u_k (np.ndarray): Matrix of directions to preserve, shape (d, k)
            
        Returns:
            np.ndarray: Oblique projection matrix of shape (d, d)
        """
        # Normalize v
        if np.abs(np.linalg.norm(v, axis=0) - 1).max() > self.eps:
            v = v / np.linalg.norm(v, axis=0, keepdims=True)
        
        # Normalize each column of u_k
        if np.abs(np.linalg.norm(u_k, axis=0) - 1).max() > self.eps:
            u_k = u_k / np.linalg.norm(u_k, axis=0, keepdims=True)
        
        # Get orthogonal complement basis of v
        V_perp_v = self.get_orthogonal_complement_basis(v)

        # Orthonormalize u_k via QR
        U_fixed, _ = np.linalg.qr(u_k)

        # compute k additional vectors from the span{u_1,...,u_k,v_1,...,v_k}
        # that is orthogonal to all of u_1, ..., u_k

        # Compute additional vector from the span{u_1,...,u_k,v}
        #U_list = [U_fixed[:, i].reshape(-1,1) for i in range(U_fixed.shape[1])]
        u_perp = self.find_orthogonal_vector_multi(U_fixed, v)

        # get an orthonormal basis of the set of all vectors orthogonal to u_perp
        V_perp_u_perp =self.get_orthogonal_complement_basis(u_perp)
     
        # get the projection matrix - first define the matrix to be inverted
        A = np.matmul(V_perp_v.T, V_perp_u_perp)
        A_inv = np.linalg.pinv(A)

        # get the projection matrix
        P = np.matmul(np.matmul(V_perp_u_perp, A_inv), V_perp_v.T)

        return P
    

    
    def find_orthogonal_vector_multi(self, U, v):
        """
        Given a list of vectors u_list and matrix v, returns k vectors that are:
        1. In the span of {u_1,...,u_k,v_1,...,v_k}
        2. Strictly orthogonal to all vectors in u_list
        3. Orthonormal to each other
        
        Args:
            u_list (list): List of vectors each of shape (d,1) representing u_1,...,u_k
            v (np.ndarray): Matrix of shape (d,k) representing v_1,...,v_k
            
        Returns:
            np.ndarray: Matrix of shape (d,k) whose columns form orthonormal vectors
        """
       
            
        # Get orthonormal basis for span of U
        Q_u, _ = np.linalg.qr(U)
        
        # Form matrix of all vectors [U V]
        UV = np.column_stack([U, v])
        
        # Get orthonormal basis for combined span
        Q_uv, _ = np.linalg.qr(UV)
        
        # For each vector in Q_uv, project out Q_u components and normalize
        k = v.shape[1]  # number of vectors we need
        result = []

        # turn U into a list of vectors
        u_list = [U[:, i:i+1] for i in range(U.shape[1])]
        
        for i in range(Q_uv.shape[1]):
            q = Q_uv[:, i:i+1]  # keep as matrix
            
            # Project out all components in U direction
            for u in u_list:
                q = q - (u.T @ q) * u
                
            # Check if vector is non-zero after projection
            norm = np.linalg.norm(q)
            if norm > self.eps:
                q = q / norm
                
                # Also ensure it's orthogonal to previously found vectors
                for r in result:
                    q = q - (r.T @ q) * r
                
                # If still non-zero after all projections, keep it    
                if np.linalg.norm(q) > self.eps:
                    result.append(q)
                    
            if len(result) == k:
                break
                
        if len(result) < k:
            raise ValueError(f"Could only find {len(result)} orthogonal vectors instead of requested {k}")
            
        return np.column_stack(result)







