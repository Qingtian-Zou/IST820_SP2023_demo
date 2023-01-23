This task is to train and evaluate the random forest model's performances.

The general steps are listed below:
- Flatten original X data. This is needed because random forest model can only deal with 1D data, but not multi-dimensional data like the 3D matrix originally provided.
    - Done with "flatten-dataset.py"
- Dimension reduction. Traditional machine learning techniques are known to work poorly when the input space is too sparse. Therefore, dimension reduction may be needed to down-size the input space.
    - Done with "pca.r"
- Model training and evaluation.
    - Done with "random-forest.r"

Please note that R codes, but not pure Python codes, are provided here. They are intended to demonstrate the workflow but not to be handed to students directly.