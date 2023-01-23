This task is to load provided data and model, and then evaluate the model's performances. A Python script "load-model.py" is provided for demonstration.

The data is stored in the "Data" directory, namely "X.npy" and "Y.npy". They are loaded at line 69 and 70 in "load-model.py". The models are stored in the "Model" directory.

The script needs 2 arguments. Note that models in the "Model" directory are named following the "classifier-n-k-[0,1,2,3].h5". "n" is the training id, "k" is the hyper-parameter set, and the last "[0,1,2,3]" represents four models from four folds in cross validation. As for the provided model, n=99 and k=0, so this script should be run with:
```
python load-model.py --n 99 --k 0
```
