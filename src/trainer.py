from keras.optimizers import Adadelta, Adam
from keras.models import Model, load_model


class Trainer:

    def __init__(self):
        pass

    def compile(self, model):
        model.compile(
            optimizer="Adam",
            loss="mse",
            metrics=['accuracy']
        )

    def train(self):
        batch_size = 32
        epochs = 1
        fpath = learned_model_path + \
            '{epoch:04d}-{loss:.2f}-{val_loss:.2f}.hdf5'
        es_cb = EarlyStopping(patience=30, verbose=0)
        mc_cb = ModelCheckpoint(filepath=fpath, monitor='val_loss',
                                verbose=0, save_best_only=True, mode='auto',  period=5)
        tb_cb = TensorBoard(log_dir=learned_model_path, histogram_freq=1)
        callbacks = []
        callbacks.append(es_cb)
        callbacks.append(mc_cb)
        callbacks.append(tb_cb)
        result = model.fit(
            x=x_train,
            y=y_train,
            batch_size=batch_size,
            #     initial_epoch = 90,
            epochs=epochs,
            validation_data=(x_valid, y_valid),
            verbose=1,
            callbacks=callbacks
        )
