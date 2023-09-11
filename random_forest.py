import numpy
from . import regression_tree_cart


class Forest(object):
    """
    the class is updated to work with numpy and dataframe alike structure
    """
    def __init__(self, trees):
        self.trees = trees

    def lookup(self, x):
        """Returns the predicted value given the parameters."""
        preds = []
        for tree in self.trees:
            preds.append(tree.lookup(x))
        return numpy.mean(preds)

    def predict_all(self, x):
        """Returns the predicted values for a list of data points."""
        preds = []
        for row in x:
            preds.append(self.lookup(row))
        return preds

    def get_loss(self):
        """
        Get the average error for a tree for all trees in the forest
        @return: Errors from individual trees
        """
        losses = []
        for tree in self.trees:
            error, number_of_nodes = tree.get_cost_params()
            losses.append(numpy.divide(error, number_of_nodes))
        return losses


def make_boot(x, y, number_of_trees):
    """Construct a bootstrap sample from the data."""
    inds = numpy.random.choice(len(x), size=numpy.floor(len(x)/number_of_trees).astype(int), replace=True)
    return x[inds], y[inds]


def make_forest(x, y, number_of_trees, max_depth=500, min_samples_split=5, labels={}, loss_function=None):
    """
    Function to generate the random forest regressor
    @param x:
    @param y:
    @param number_of_trees:
    @param max_depth:
    @param min_samples_split:
    @param labels:
    @param loss_function: Custom loss function
    @return:
    """
    if len(x) <= min_samples_split * number_of_trees:
        raise ValueError('Size of input should be lager than min_branch_size * min_samples_split')

    """Function to grow a random forest given some training data."""
    trees = []
    for b in range(number_of_trees):
        subsample_x, subsample_y = make_boot(x, y, number_of_trees)
        trees.append(
            regression_tree_cart.grow_tree(
                subsample_x,
                subsample_y,
                0,
                max_depth=max_depth,
                min_samples_split=min_samples_split,
                labels=labels,
                start=True,
                feat_bag=True,
                loss_function=loss_function
            )
        )
    print('Forest generated')

    return Forest(trees)
