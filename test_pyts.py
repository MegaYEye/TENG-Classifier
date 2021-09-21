from pyts.datasets import load_gunpoint
from pyts.utils import windowed_view

# Load the data set and fit the classifier
X, _, y, _ = load_gunpoint(return_X_y=True)
# print(X)

from pyts.datasets import make_cylinder_bell_funnel


X, y = make_cylinder_bell_funnel(n_samples=12, random_state=42)
print(X.shape)