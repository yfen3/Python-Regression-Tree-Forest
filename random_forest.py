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


def make_boot(pairs, n):
    """Construct a bootstrap sample from the data."""
    inds = numpy.random.choice(n, size=n, replace=True)
    return dict(map(lambda x: pairs[x], inds))


def make_forest(x, y, number_of_trees, max_depth=500, Nmin=5, labels={}, loss_function=None):
    """Function to grow a random forest given some training data."""
    trees = []
    #n = len(data)
    #pairs = data.items()
    for b in range(number_of_trees):
        #boot = make_boot(pairs, n)
        trees.append(
            regression_tree_cart.grow_tree(
                x,
                y,
                0,
                max_depth=max_depth,
                Nmin=Nmin,
                labels=labels,
                start=True,
                feat_bag=True,
                loss_function=loss_function
            )
        )
    print('Forest generated')

    return Forest(trees)
