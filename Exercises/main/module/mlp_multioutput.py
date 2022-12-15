from keras.models import Sequential
from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras.layers import Dense

def mlp_multioutput(X_train, y_train, X_valid, y_valid, LATENT_DIM, BATCH_SIZE, EPOCHS, LAG, HORIZON, 
                    loss, optimizer, earlystop, best_val, verbose):
  
    model = Sequential()
    model.add(Dense(LATENT_DIM, activation="relu", input_shape=(LAG,)))
    model.add(Dense(HORIZON))

    model.compile(optimizer=optimizer, loss=loss)
    
    history = model.fit(X_train,
                        y_train,
                        batch_size=BATCH_SIZE,
                        epochs=EPOCHS,
                        validation_data=(X_valid, y_valid),
                        callbacks=[earlystop, best_val],
                        verbose=verbose)
    return model, history
