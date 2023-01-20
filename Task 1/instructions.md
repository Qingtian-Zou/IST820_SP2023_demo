This task is to load provided data and model, and then evaluate the model's performances. A Python script "load-model.py" is provided for demonstration.

The data is stored in the "Data" directory, namely "X.npy" and "Y.npy". They are loaded at line 69 and 70 in "load-model.py". The models are stored in the "Model" directory.

The script needs 2 arguments. Note that models in the "Model" directory are named following the "classifier-n-k-[1,2,3,4].h5". "n" is the training id, "k" is the hyper-parameter set, and the last "[1,2,3,4]" represents four models from four folds in cross validation. In the provided example, n=2 and k=0, so this script should be ran with:
```
python load-model.py --n 2 --k 0
```
