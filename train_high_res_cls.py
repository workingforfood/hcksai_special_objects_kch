import os
import pickle

import numpy as np
import tensorflow as tf
from keras.callbacks import ModelCheckpoint

from data_generator import Classifier_DataGenerator
from models_base import efficientnet_classifierB2
from net_utils import ReduceLROnPlateau

np.random.seed(54321)
tf.set_random_seed(54321)


if __name__=="__main__":

    model_name = "efB2_balanced_ce_RE"
    data_path = "/var/data2/datasets/hacksai/train"
    num_epochs = 40

    os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    base_save = "chkpts_{}".format(model_name)
    filename = base_save + "/base_{}".format(model_name) + "_{epoch}_{acc}.h5"

    if not os.path.exists(base_save):
        os.mkdir(base_save)

    with open("train_list.pkl", "rb") as fp:
        train_data = pickle.load(fp)

    # balance classes
    extend_calasses = []
    expander_data = {0: 1, 1: 0, 2: 2, 3: 0, 4: 1, 5: 1, 6: 2, 7: 2}
    for sample in train_data:
        for i in range(expander_data[sample["label"]]):
            extend_calasses.append(sample)
    train_data = train_data + extend_calasses

    train_gen = Classifier_DataGenerator(data_path, train_data, img_shape=(224, 224), batch_size=32,
                                         balance_classes=True, shuffle=True, istrain=True)

    model = efficientnet_classifierB2(input_shape=(224, 224, 3), num_of_cls=8)
    model.summary()

    chkpcallback = ModelCheckpoint(filename, verbose=1, period=1, save_best_only=False)
    reduce_lr = ReduceLROnPlateau(factor=0.5, patience=1, verbose=1, min_lr=1e-7,
                                  min_delta=[1e-2, 1e-3, 1e-4, 1e-5], monitor="loss", mode='min')

    model.fit_generator(train_gen,
                        use_multiprocessing=True,
                        workers=4,
                        epochs=num_epochs,
                        max_queue_size=10,
                        callbacks=[chkpcallback, reduce_lr],
                        initial_epoch=0)
    model.save(base_save + "/final_{}.h5".format(model_name))


