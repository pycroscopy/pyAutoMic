
import numpy as np

try:
    import torch
    from botorch.models import SingleTaskGP
    from botorch.fit import fit_gpytorch_mll
    from botorch.acquisition import ExpectedImprovement
    from botorch.optim import optimize_acqf
    from gpytorch.mlls import ExactMarginalLogLikelihood
    DEPENDENCIES_INSTALLED = True
except ImportError:
    DEPENDENCIES_INSTALLED = False

# TODO: add: discretizations of x values
#       add: min or max -- sign : goal is for non-programmers
#       add: scaling the data dfed to model b/w 0 to 1
#       minim comes from the minim of predictive distribution
class Tune1d:
    def __init__(self, variable, python_command, num_gp_steps, bounds, seed_pt=None):
        if not DEPENDENCIES_INSTALLED:
            raise ImportError("Required dependencies (torch, botorch, gpytorch) are not installed. "
                              "Please install them to use this module.")
        self.variable = variable
        self.python_command = python_command
        self.num_gp_steps = num_gp_steps
        self.bounds = bounds
        self.seed_pt = seed_pt
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        if seed_pt is not None:
            torch.manual_seed(seed_pt)
            np.random.seed(seed_pt)

    def _acquire_data(self, x):
        result = self.python_command(x.item())
        return result


    def _initialize_model(self, train_x, train_y):
        train_x = train_x.to(self.device)
        train_y = train_y.to(self.device)
        model = SingleTaskGP(train_x, train_y).to(self.device)
        mll = ExactMarginalLogLikelihood(model.likelihood, model)
        fit_gpytorch_mll(mll)
        return model

    def _optimize_acquisition_function(self, model, train_y, bounds):
        EI = ExpectedImprovement(model, best_f=train_y.max().item())
        candidate, _ = optimize_acqf(
            EI, bounds=bounds, q=1, num_restarts=10, raw_samples=20,
        )
        return candidate

    def optimize(self):
        # Ensure that train_x and train_y are of the same dtype (e.g., float32)
        dtype = torch.float32
        train_x = (torch.rand(1, 1, dtype=dtype, device=self.device) * (self.bounds[1, 0] - self.bounds[0, 0]) + self.bounds[0, 0]).to(self.device)
        train_y = torch.tensor([[self._acquire_data(train_x[0])]], dtype=dtype).to(self.device)

        for _ in range(self.num_gp_steps):
            model = self._initialize_model(train_x, train_y)
            new_x = self._optimize_acquisition_function(model, train_y, self.bounds).to(self.device)
            new_y = torch.tensor([[self._acquire_data(new_x[0])]], dtype=dtype).to(self.device)
            
            train_x = torch.cat([train_x, new_x])
            train_y = torch.cat([train_y, new_y])

        best_idx = train_y.argmax()
        return train_x[best_idx].item(), train_y[best_idx].item()

# if __name__ == "__main__":

#     import numpy as np
#     import os
#     # os.path.append("")
#     from BOtune.tune import Tune
#     import torch

#     def generate_noisy_image(x):
#         # Generate a 10x10 image with noise
#         image = np.random.normal(0, 1, (10, 10))
#         # Add a signal that depends on x
#         image += x * np.sin(np.arange(100).reshape(10, 10) / 10)
#         return image

#     # Define the optimization problem
#     variable = 'x'
#     python_command = generate_noisy_image
#     num_gp_steps = 20
#     bounds = torch.tensor([[0.0], [1.0]])  # Shape is [2, 1]--> range in which search for optimal value

#     # Create and run the optimizer
#     optimizer = Tune(variable, python_command, num_gp_steps, bounds, seed_pt=42)
#     best_x, best_y = optimizer.optimize()

#     print(f"Best x: {best_x}")
#     print(f"Best y (std dev): {best_y}")

import numpy as np

try:
    import torch
    from botorch.models import SingleTaskGP
    from botorch.fit import fit_gpytorch_mll
    from botorch.acquisition import ExpectedImprovement
    from botorch.optim import optimize_acqf
    from gpytorch.mlls import ExactMarginalLogLikelihood
    DEPENDENCIES_INSTALLED = True
except ImportError:
    DEPENDENCIES_INSTALLED = False

# TODO: add: discretizations of x values
#       add: min or max -- sign : goal is for non-programmers
class Tune2d:
    def __init__(self, variable, python_command, num_gp_steps, bounds, seed_pt=None):
        if not DEPENDENCIES_INSTALLED:
            raise ImportError("Required dependencies (torch, botorch, gpytorch) are not installed. "
                              "Please install them to use this module.")
        self.variable = variable
        self.python_command = python_command
        self.num_gp_steps = num_gp_steps
        self.bounds = bounds
        self.seed_pt = seed_pt
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        if seed_pt is not None:
            torch.manual_seed(seed_pt)
            np.random.seed(seed_pt)

    def _acquire_data(self, x):
        result = self.python_command(x.cpu().numpy())  # Convert tensor to numpy array before passing to the function
        return result

    def _initialize_model(self, train_x, train_y):
        train_x = train_x.to(self.device)
        train_y = train_y.to(self.device)
        model = SingleTaskGP(train_x, train_y).to(self.device)
        mll = ExactMarginalLogLikelihood(model.likelihood, model)
        fit_gpytorch_mll(mll)
        return model

    def _optimize_acquisition_function(self, model, train_y, bounds):
        EI = ExpectedImprovement(model, best_f=train_y.max().item())
        candidate, _ = optimize_acqf(
            EI, bounds=bounds, q=1, num_restarts=10, raw_samples=20,
        )
        return candidate

    def optimize(self):
        # Ensure that train_x and train_y are of the same dtype (e.g., float32)
        dtype = torch.float32
        train_x = (torch.rand(1, 2, dtype=dtype, device=self.device) * (self.bounds[1] - self.bounds[0]) + self.bounds[0]).to(self.device)
        train_y = torch.tensor([[self._acquire_data(train_x[0])]], dtype=dtype).to(self.device)


        for _ in range(self.num_gp_steps):
            model = self._initialize_model(train_x, train_y)
            new_x = self._optimize_acquisition_function(model, train_y, self.bounds).to(self.device)
            new_y = torch.tensor([[self._acquire_data(new_x[0])]], dtype=dtype).to(self.device)
            
            train_x = torch.cat([train_x, new_x])
            train_y = torch.cat([train_y, new_y])

        best_idx = train_y.argmax()
        return train_x[best_idx].tolist(), train_y[best_idx].item()

# if __name__ == "__main__":

#     import numpy as np
#     import os
#     import torch

#     def generate_noisy_image(x):
#         # Generate a 10x10 image with noise
#         image = np.random.normal(0, 1, (10, 10))
#         # Add a signal that depends on x
#         x1, x2 = x
#         image += x1 * np.sin(np.arange(100).reshape(10, 10) / 10)
#         image += x2 * np.cos(np.arange(100).reshape(10, 10) / 10)
#         return np.std(image)  # Return a scalar value to optimize

#     # Define the optimization problem
#     variable = 'x'
#     python_command = generate_noisy_image
#     num_gp_steps = 20
#     bounds = torch.tensor([[0.0, 0.0], [1.0, 1.0]])  # Shape is [2, 2]--> range in which search for optimal value for both dimensions

#     # Create and run the optimizer
#     optimizer = Tune2d(variable, python_command, num_gp_steps, bounds, seed_pt=42)
#     best_x, best_y = optimizer.optimize()

#     print(f"Best x: {best_x}")
#     print(f"Best y (std dev): {best_y}")

import numpy as np

try:
    import torch
    from botorch.models import SingleTaskGP
    from botorch.fit import fit_gpytorch_mll
    from botorch.acquisition import ExpectedImprovement
    from botorch.optim import optimize_acqf
    from gpytorch.mlls import ExactMarginalLogLikelihood
    DEPENDENCIES_INSTALLED = True
except ImportError:
    DEPENDENCIES_INSTALLED = False

class Tune3d:
    def __init__(self, variable, python_command, num_gp_steps, bounds, seed_pt=None):
        if not DEPENDENCIES_INSTALLED:
            raise ImportError("Required dependencies (torch, botorch, gpytorch) are not installed. "
                              "Please install them to use this module.")
        self.variable = variable
        self.python_command = python_command
        self.num_gp_steps = num_gp_steps
        self.bounds = bounds
        self.seed_pt = seed_pt
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        if seed_pt is not None:
            torch.manual_seed(seed_pt)
            np.random.seed(seed_pt)

    def _acquire_data(self, x):
        # Convert the tensor to numpy array before passing to the function
        result = self.python_command(x.cpu().numpy())
        return result

    def _initialize_model(self, train_x, train_y):
        train_x = train_x.to(self.device)
        train_y = train_y.to(self.device)
        model = SingleTaskGP(train_x, train_y).to(self.device)
        mll = ExactMarginalLogLikelihood(model.likelihood, model)
        fit_gpytorch_mll(mll)
        return model

    def _optimize_acquisition_function(self, model, train_y, bounds):
        EI = ExpectedImprovement(model, best_f=train_y.max().item())
        candidate, _ = optimize_acqf(
            EI, bounds=bounds, q=1, num_restarts=10, raw_samples=20,
        )
        return candidate

    def optimize(self):
        # Ensure that train_x and train_y are of the same dtype (e.g., float32)
        dtype = torch.float32
        train_x = (torch.rand(1, 3, dtype=dtype, device=self.device) * (self.bounds[1] - self.bounds[0]) + self.bounds[0]).to(self.device)
        train_y = torch.tensor([[self._acquire_data(train_x[0])]], dtype=dtype).to(self.device)

        for _ in range(self.num_gp_steps):
            model = self._initialize_model(train_x, train_y)
            new_x = self._optimize_acquisition_function(model, train_y, self.bounds).to(self.device)
            new_y = torch.tensor([[self._acquire_data(new_x[0])]], dtype=dtype).to(self.device)
            
            train_x = torch.cat([train_x, new_x])
            train_y = torch.cat([train_y, new_y])

        best_idx = train_y.argmax()
        return train_x[best_idx].tolist(), train_y[best_idx].item()

# if __name__ == "__main__":

#     import numpy as np
#     import os
#     import torch

#     def generate_noisy_image(x):
#         # Generate a 10x10 image with noise
#         image = np.random.normal(0, 1, (10, 10))
#         # Unpack x
#         x1, x2, x3 = x  # x should now be a numpy array with three elements
#         image += x1 * np.sin(np.arange(100).reshape(10, 10) / 10)
#         image += x2 * np.cos(np.arange(100).reshape(10, 10) / 10)
#         image += x3 * np.tan(np.arange(100).reshape(10, 10) / 20)
#         return np.std(image)  # Return a scalar value to optimize

#     # Define the optimization problem
#     variable = 'x'
#     python_command = generate_noisy_image
#     num_gp_steps = 20
#     bounds = torch.tensor([[0.0, 0.0, 0.0], [1.0, 1.0, 1.0]])  # Shape is [2, 3] --> range in which search for optimal values for all three dimensions

#     # Create and run the optimizer
#     optimizer = Tune3d(variable, python_command, num_gp_steps, bounds, seed_pt=42)
#     best_x, best_y = optimizer.optimize()

#     print(f"Best x: {best_x}")
#     print(f"Best y (std dev): {best_y}")
