import os
import sys

import numpy as np
import tensorflow as tf
from tensorflow_addons.metrics import F1Score
from sklearn.utils import shuffle

WINDOW_SIZE = 6


def perf_measure(y_actual, y_pred):
    TP = 0
    FP = 0
    TN = 0
    FN = 0
    for i in range(len(y_pred)):
        if y_actual[i][1] == 1 and y_pred[i][1] >= 0.5:
            TP += 1
        if y_pred[i][1] >= 0.5 and y_actual[i][1] == 0:
            FP += 1
        if y_actual[i][1] == 0 and y_pred[i][1] < 0.5:
            TN += 1
        if y_pred[i][1] <= 0.5 and y_actual[i][1] == 1:
            FN += 1

    return(TP, FP, TN, FN)


def build_model(hidden_tensor, kernel_size, dropout_rate):
    inputs = tf.keras.layers.Input(shape=(WINDOW_SIZE, 32,8), name='Input')

    y = tf.keras.layers.Conv2D(filters=hidden_tensor[0], kernel_size=kernel_size[0], padding='same', activation='relu', name='Hidden0')(inputs)
    y = tf.keras.layers.MaxPooling2D(pool_size=(4, 4), strides=(3, 3), padding='same', name='Pooling0')(y)
    y = tf.keras.layers.Flatten(name='Flatten')(y)
    y = tf.keras.layers.Dense(hidden_tensor[1], activation='relu', name='Dense')(y)
    y = tf.keras.layers.Dropout(dropout_rate, name='Dropout')(y)

    probs = tf.keras.layers.Dense(2, activation='softmax', name='Output')(y)

    model = tf.keras.models.Model(inputs, probs, name='classifier4')
    return model


if __name__ == "__main__":
    gpus = tf.config.list_physical_devices('GPU')
    if gpus:
        try:
            # Currently, memory growth needs to be the same across GPUs
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
            logical_gpus = tf.config.experimental.list_logical_devices('GPU')
            print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs")
        except RuntimeError as e:
            # Memory growth must be set before GPUs have been initialized
            print(e)
    else:
        tf.config.threading.set_inter_op_parallelism_threads(8)
        tf.config.threading.set_intra_op_parallelism_threads(8)

    # training epochs and batch size
    n_epoch = 3
    n_batch = 50

    # model hyper-parameters to tune
    hidden_tensors = [[4,4]]
    kernel_sizes = [[(3, 3)]]
    dropout_rates = [0.25]

    # open file in text mode for writing training log
    n = 0
    while(os.path.isfile(os.path.join("..","Model", "training-log-"+('%.2d' % (n))+".txt"))):
        n += 1
    fi = open(os.path.join("..","Model","training-log-"+('%.2d' % (n))+".txt"), 'w')

    # record output direction
    # fiout means output to log file
    # sys.stdout means output to console
    fiout = fi
    oldStdOut = sys.stdout

    k = 0
    # by the following line, the console output is directed to be written to the log file
    sys.stdout = fiout

    # load data
    X=np.load(os.path.join("..","Data","X.npy"),allow_pickle=True)
    Y=np.load(os.path.join("..","Data","Y.npy"),allow_pickle=True)

    # shuffle loaded data, and split training and test set by 4:1
    X, Y = shuffle(X, Y, random_state=1)
    X_train=X[:4000]
    X_test=X[4000:5000]
    Y_train=Y[:4000]
    Y_test=Y[4000:5000]

    for hidden_tensor in hidden_tensors:
        for kernel_size in kernel_sizes:
            for dropout_rate in dropout_rates:
                # log training info
                fi.writelines([
                    "*********************************************************************************************************************************************\r",
                    "model parameter set "+str(k)+"\r",
                    "\r"
                ])  # "\n" is automatically append to the end of each line

                # NN: 4-fold cross validation
                acc = []
                TPs = []
                FPs = []
                TNs = []
                FNs = []
                for i in range(4):
                    print(
                        "################################################################################################################################", end="\n")
                    print("fold "+str(i+1)+"/"+str(4), end="\n")
                    # prepare data
                    tmp_train_x = X_train.tolist()
                    tmp_train_y = Y_train.tolist()
                    start = i*len(X_train)//4
                    end = (i+1)*len(X_train)//4
                    del(tmp_train_x[start:end])
                    del(tmp_train_y[start:end])
                    tmp_train_x = np.array(tmp_train_x)
                    tmp_train_y = np.array(tmp_train_y)
                    tmp_val_x = X_train[start:end]
                    tmp_val_y = Y_train[start:end]

                    # build model
                    model = build_model(hidden_tensor, kernel_size, dropout_rate)
                    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=[F1Score(num_classes=2)])
                    print(model.summary())

                    # start training
                    model.fit(tmp_train_x, tmp_train_y, validation_data=(tmp_val_x, tmp_val_y), epochs=n_epoch, batch_size=n_batch)

                    # evaluate
                    scores = model.evaluate(X_test, Y_test, batch_size=n_batch, verbose=0)
                    Y_predict = model.predict(X_test,batch_size=n_batch)
                    Mat = perf_measure(Y_test, Y_predict)
                    acc.append(scores[1][0])

                    sys.stdout = oldStdOut
                    print("Model f1 score: %.3f" % (scores[1][0]*100))
                    print(Mat)
                    print()
                    sys.stdout = fiout
                    print("Model f1 score: %.3f" % (scores[1][0]*100))
                    print("(TP, FP, TN, FN)"+str(Mat))
                    model.save(os.path.join("..","Model", "classifier-"+str(n)+"-"+str(k)+"-"+str(i)+".h5"))
                    TPs.append(Mat[0])
                    FPs.append(Mat[1])
                    TNs.append(Mat[2])
                    FNs.append(Mat[3])

                # summarize performance across four folds
                print("######################################################################################################", end="\n")
                print("all folds done", end="\n")
                print("average f1 score: %.3f" %
                        (100*sum(acc)/len(acc)), end="\n")
                print("average confusion matrix (TP,FP,TN.FN): (%.2f, %.2f, %.2f, %.2f)" % (
                    (sum(TPs)/len(TPs)), (sum(FPs)/len(FPs)), (sum(TNs)/len(TNs)), (sum(FNs)/len(FNs))))
                print("average false positive rate: %.3f%%" %
                        (100*sum(FPs)/(sum(FPs)+sum(TNs))))
                print("average false negative rate: %.3f%%" %
                        (100*sum(FNs)/(sum(FNs)+sum(TPs))))
                sys.stdout = oldStdOut
                print("Completed %d/%s" % (k+1, len(hidden_tensors)*len(kernel_sizes)*len(dropout_rates)))
                sys.stdout = fiout
                k += 1

sys.stdout = oldStdOut
fi.close()
