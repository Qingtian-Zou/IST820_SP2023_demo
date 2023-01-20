import os
import argparse

import numpy as np
import tensorflow as tf
from tensorflow_addons.metrics import F1Score

FLAGS=None

# performance measurement function
def perf_measure(y_actual, y_pred):
    # true positive: malicious data sample predicted as malicious
    TP = 0
    # false positive: benign data sample predicted as malicious
    FP = 0
    # true negative: benign data sample predicted as benign
    TN = 0
    # false negative: malicious data sample predicted as benign
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


if __name__ == "__main__":
    # parse arguments from command line
    parser=argparse.ArgumentParser()
    parser.add_argument(
        '--n',
        type=int,
        help='ID of training.'
    )
    parser.add_argument(
        '--k',
        type=int,
        help='ID of model. The models to load should be placed in the model directory, named \"classifier-N-K-[0,1,2,3].h5\"'
    )
    FLAGS, unparsed =parser.parse_known_args()

    # Test if GPU acceleration is available.
    # If so, set memory occupation mode to incremental, so that TensorFlow will occupy video RAM as needed.
    # Otherwise, TensorFlow will occupy more than 90% video RAM at once.
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

    # load data
    X=np.load(os.path.join("..","Data","X.npy"),allow_pickle=True)
    Y=np.load(os.path.join("..","Data","Y.npy"),allow_pickle=True)

    # load four models
    # for every model, make it predict on provided data, and record prediction results
    predictions=[]
    for i in range(4):
        model=tf.keras.models.load_model(os.path.join("..","Model","classifier-"+str(FLAGS.n)+"-"+str(FLAGS.k)+"-"+str(i)+".h5"),custom_objects={"metric":F1Score(num_classes=2)})
        predictions.append(model.predict(X))

    # calculate average prediction results all four models' outputs
    pred=np.average(predictions,axis=0)

    # print out model summary
    model.summary()

    # calculate and print out evaluation metrics
    Mat=perf_measure(Y,pred)
    print("Model accuracy: %.2f %%" % ((Mat[0]+Mat[2])/sum(Mat)*100))
    print("Model f1 score: %.4f" % (Mat[0]/(Mat[0]+0.5*(Mat[1]+Mat[3]))))
    print("Model detection rate: %.2f %%" % (Mat[0]/(Mat[0]+Mat[3])*100))
    if Mat[1]+Mat[2]==0:
        print("FPR NaN!")
    else:
        print("Model FPR: %.2f %%\n" % (Mat[1]/(Mat[1]+Mat[2])*100))
    print("(TP, FP, TN, FN):"+str(Mat))
