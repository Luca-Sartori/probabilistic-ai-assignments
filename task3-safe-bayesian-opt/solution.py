"""Solution."""
import numpy as np
from scipy.optimize import fmin_l_bfgs_b
from sklearn.gaussian_process.kernels import Matern, DotProduct, ConstantKernel as C, RBF
from sklearn.gaussian_process import GaussianProcessRegressor
from scipy.stats import norm
# import additional ...


# global variables
DOMAIN = np.array([[0, 10]])  # restrict \theta in [0, 10]
SAFETY_THRESHOLD = 4  # threshold, upper bound of SA

class BO_algo():
    def __init__(self):
        """Initializes the algorithm with a parameter configuration."""

        self.kernel_f = C(1.0) * Matern(length_scale=0.5, nu=2.5)
        self.kernel_v = C(np.sqrt(2), constant_value_bounds="fixed") * RBF(length_scale=0.5, length_scale_bounds="fixed") + C(1.0) * Matern(length_scale=0.5, nu=2.5)

        self.beta = 2
        self.gamma = 1
        self.lamda = 5

        self.sigma_f = 0.15
        self.sigma_v = 0.0001

        self.X_observed = []
        self.y_f_observed = []
        self.y_v_observed = []

        self.prior_mean_v = 4.0

        self.best_x = None
        
        self.initial_safe_point = None
        self.safe_bounds = np.array([[None, None]])

    def update_safe_bounds(self):
        """Update safe bounds based on GP model of v."""
        
        # set initial safe bounds
        self.safe_bounds = np.array([[self.initial_safe_point, self.initial_safe_point]])
        
        # compute v predictions on a grid
        x_grid = np.linspace(DOMAIN[0, 0], DOMAIN[0, 1], 1000).reshape(-1, 1)
        mu_v, std_v = self.v.predict(x_grid, return_std=True)
        upper_bound = mu_v + self.gamma * std_v + self.prior_mean_v
        
        # find safe bounds by moving left and right from initial safe point
        init_idx = np.argmin(np.abs(x_grid - self.initial_safe_point))
        # move left
        found_left = False
        for i in range(init_idx, -1, -1):
            if upper_bound[i] > SAFETY_THRESHOLD:
                self.safe_bounds[0, 0] = x_grid[i + 1] if i + 1 < len(x_grid) else x_grid[i]
                found_left = True
                break
            
        # move right
        found_right = False
        for i in range(init_idx, len(x_grid)):
            if upper_bound[i] > SAFETY_THRESHOLD:
                self.safe_bounds[0, 1] = x_grid[i - 1] if i - 1 >= 0 else x_grid[i]
                found_right = True
                break
            
        if not found_left:
            self.safe_bounds[0, 0] = DOMAIN[0, 0]
        if not found_right:
            self.safe_bounds[0, 1] = DOMAIN[0, 1]

    def next_recommendation(self):
        """
        Recommend the next input to sample.

        Returns
        -------
        recommendation: np.ndarray
            the next point to evaluate
        """
        
        # if first observation, print initial point info, and set safe bounds
        if len(self.X_observed) == 1:
            print(f'\n\nInitial safe point: x = {self.X_observed[0]}, f(x) = {self.y_f_observed[0]}, v(x) = {self.y_v_observed[0]}')        

        X = np.array(self.X_observed).reshape(-1, 1)
        y_f = np.array(self.y_f_observed).reshape(-1, 1)
        y_v = np.array(self.y_v_observed).reshape(-1, 1) - self.prior_mean_v

        self.f = GaussianProcessRegressor(kernel=self.kernel_f, alpha=self.sigma_f ** 2, optimizer=None, normalize_y=False)
        self.v = GaussianProcessRegressor(kernel=self.kernel_v, alpha=self.sigma_v ** 2, optimizer=None, normalize_y=False)

        self.f.fit(X, y_f)
        self.v.fit(X, y_v)
        
        # update safe bounds by moving left and right until unsafe
        self.update_safe_bounds()

        reccomendation = self.safe_optimize_acquisition_function()
        
        
        print(f'Recommended next point: x = {reccomendation:.2f},\tacquisition value = {float(self.acquisition_function(reccomendation)):.2f},\t safe bounds = {float(self.safe_bounds[0, 0]):.3f} to {float(self.safe_bounds[0, 1]):.3f}')
        
        return reccomendation

    def optimize_acquisition_function(self):
        """Optimizes the acquisition function defined below (DO NOT MODIFY).

        Returns
        -------
        x_opt: float
            the point that maximizes the acquisition function, where
            x_opt in range of DOMAIN
        """

        def objective(x):
            return -self.acquisition_function(x)

        f_values = []
        x_values = []

        # Restarts the optimization 20 times and pick the best solution
        for _ in range(20):
            x0 = DOMAIN[:, 0] + (DOMAIN[:, 1] - DOMAIN[:, 0]) * \
                 np.random.rand(DOMAIN.shape[0])
            result = fmin_l_bfgs_b(objective, x0=x0, bounds=DOMAIN,
                                   approx_grad=True)
            x_values.append(np.clip(result[0], *DOMAIN[0]))
            f_values.append(-result[1])

        ind = np.argmax(f_values)
        x_opt = x_values[ind].item()
        # print(f'x_opt = {x_opt}')
        return x_opt

    def safe_optimize_acquisition_function(self):
        """Optimizes the acquisition function within safe bounds.

        Returns
        -------
        x_opt: float
            the point that maximizes the acquisition function within safe bounds
        """
        
        def objective(x):
            return -self.acquisition_function(x)

        f_values = []
        x_values = []

        # Restarts the optimization 20 times and pick the best solution
        for _ in range(20):
            x0 = self.safe_bounds[:, 0] + (self.safe_bounds[:, 1] - self.safe_bounds[:, 0]) * \
                 np.random.rand(self.safe_bounds.shape[0])
            result = fmin_l_bfgs_b(objective, x0=x0, bounds=self.safe_bounds,
                                   approx_grad=True)
            x_values.append(np.clip(result[0], *self.safe_bounds[0]))
            f_values.append(-result[1])

        ind = np.argmax(f_values)
        x_opt = x_values[ind].item()
        return x_opt

    def acquisition_function(self, x: np.ndarray):
        """Compute the acquisition function for x.

        Parameters
        ----------
        x: np.ndarray
            x in domain of f, has shape (N, 1)

        Returns
        ------
        af_value: np.ndarray
            shape (N, 1)
            Value of the acquisition function at x
        """
        x = np.atleast_2d(x)
        
        mu_f, std_f = self.f.predict(x, return_std=True)
        
        mu_v, std_v = self.v.predict(x, return_std=True)
        mu_v = mu_v + self.prior_mean_v
        prob_safe = norm.cdf((SAFETY_THRESHOLD - mu_v) / std_v)
        
        return (mu_f + self.beta * std_f)  - self.lamda * np.maximum(mu_v + 2 * std_v - SAFETY_THRESHOLD, 0)
        
        


    def add_data_point(self, x: float, f: float, v: float):
        """
        Add data points to the model.

        Parameters
        ----------
        x: float or np.ndarray
            structural features
        f: float
            logP obj func
        v: float
            SA constraint func
        """
    
        # Store the initial safe point
        if self.initial_safe_point is None:
            self.initial_safe_point = x

        self.X_observed.append(x)
        self.y_f_observed.append(f)
        self.y_v_observed.append(v)
        
        # print warning if unsafe point is added
        if v >= SAFETY_THRESHOLD:
            print(f'↑↑↑↑ WARNING: Unsafe point added! x = {x}, v(x) = {v} ↑↑↑↑')
        

    def get_solution(self):
        """
        Return x_opt that is believed to be the maximizer of f.

        Returns
        -------
        solution: float
            the optimal solution of the problem
        """
        
        # best_f = -np.inf
        # best_x = None

        # for i, (x, f_val, v_val) in enumerate(zip(
        #         self.X_observed, self.y_f_observed, self.y_v_observed
        # )):
        #     if v_val < SAFETY_THRESHOLD and f_val > best_f:
        #         best_f = f_val
        #         best_x = x

        # if best_x is None:
        #     min_v_idx = np.argmin(self.y_v_observed)
        #     best_x = self.X_observed[min_v_idx]

        # return best_x
        
        # Use the GP model to predict the best safe point
        X = np.array(self.X_observed).reshape(-1, 1)
        y_f = np.array(self.y_f_observed).reshape(-1, 1)
        y_v = np.array(self.y_v_observed).reshape(-1, 1)
        
        self.f = GaussianProcessRegressor(kernel=self.kernel_f, alpha=self.sigma_f ** 2, optimizer=None, normalize_y=False)
        self.v = GaussianProcessRegressor(kernel=self.kernel_v, alpha=self.sigma_v ** 2, optimizer=None, normalize_y=False)
        
        self.f.fit(X, y_f)
        self.v.fit(X, y_v - self.prior_mean_v)
        
        # get safe bounds
        self.update_safe_bounds()
        
        # optimize f within safe bounds
        def objective(x):
            return -self.f.predict(x.reshape(1, -1))
        
        f_values = []
        x_values = []
        
        # Restarts the optimization 20 times and pick the best solution
        for _ in range(20):
            x0 = self.safe_bounds[:, 0] + (self.safe_bounds[:, 1] - self.safe_bounds[:, 0]) * \
                 np.random.rand(self.safe_bounds.shape[0])
            result = fmin_l_bfgs_b(objective, x0=x0, bounds=self.safe_bounds,
                                   approx_grad=True)
            x_values.append(np.clip(result[0], *self.safe_bounds[0]))
            f_values.append(-result[1])
        ind = np.argmax(f_values)
        x_opt = x_values[ind].item()
        return x_opt
# ---
# TOY PROBLEM. To check your code works as expected (ignored by checker).
# ---

def check_in_domain(x: float):
    """Validate input"""
    x = np.atleast_2d(x)
    return np.all(x >= DOMAIN[None, :, 0]) and np.all(x <= DOMAIN[None, :, 1])


def f(x: float):
    """Dummy logP objective"""
    mid_point = DOMAIN[:, 0] + 0.5 * (DOMAIN[:, 1] - DOMAIN[:, 0])
    return - np.linalg.norm(x - mid_point, 2)


def v(x: float):
    """Dummy SA"""
    return 2.0


def get_initial_safe_point():
    """Return initial safe point"""
    x_domain = np.linspace(*DOMAIN[0], 4000)[:, None]
    c_val = np.vectorize(v)(x_domain)
    x_valid = x_domain[c_val < SAFETY_THRESHOLD]
    np.random.seed(0)
    np.random.shuffle(x_valid)
    x_init = x_valid[0]

    return x_init


def main():
    """FOR ILLUSTRATION / TESTING ONLY (NOT CALLED BY CHECKER)."""
    # Init problem
    agent = BO_algo()

    # Add initial safe point
    x_init = get_initial_safe_point()
    obj_val = f(x_init)
    cost_val = v(x_init)
    agent.add_data_point(x_init, obj_val, cost_val)

    # Loop until budget is exhausted
    for j in range(20):
        # Get next recommendation
        x = agent.next_recommendation()

        # Check for valid shape
        assert x.shape == (1, DOMAIN.shape[0]), \
            f"The function next recommendation must return a numpy array of " \
            f"shape (1, {DOMAIN.shape[0]})"

        # Obtain objective and constraint observation
        obj_val = f(x) + np.randn()
        cost_val = v(x) + np.randn()
        agent.add_data_point(x, obj_val, cost_val)

    # Validate solution
    solution = agent.get_solution()
    assert check_in_domain(solution), \
        f'The function get solution must return a point within the' \
        f'DOMAIN, {solution} returned instead'

    # Compute regret
    regret = (0 - f(solution))

    print(f'Optimal value: 0\nProposed solution {solution}\nSolution value '
          f'{f(solution)}\nRegret {regret}\nUnsafe-evals TODO\n')


if __name__ == "__main__":
    main()
