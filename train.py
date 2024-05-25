import tensorflow as tf
import numpy as np
from train_data_humain import inputs_data, sortie_prevue
from test_data_humain import sortie_prevue_test, testData




IS_BIASED = True  


data_entre = np.array(inputs_data)
sortie = np.array(sortie_prevue)
new_data_x = np.array(testData)
new_data_y = np.array(sortie_prevue_test)


# Créer le modèle
model = tf.keras.Sequential([
    tf.keras.layers.Input(shape=(754,)), # Input layer (no bias) 
    tf.keras.layers.Dense(15, activation='sigmoid',use_bias=IS_BIASED),  
    tf.keras.layers.Dense(4, activation='sigmoid',use_bias=IS_BIASED),   
    tf.keras.layers.Dense(1, activation='sigmoid' ,use_bias=IS_BIASED)  
])


# Compiler le modèle
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# Entraîner le modèle
model.fit(data_entre, sortie, epochs=1000)


 # Fonction pour préparer vos nouvelles données
loss, accuracy = model.evaluate(new_data_x, new_data_y)
#afficher la sortie du reseau
print("Sortie du réseau :", model.predict(new_data_x))
print("Sortie prévue :", new_data_y)


print("Perte :", loss)
print("Précision :", accuracy)


weights_biases = model.get_weights()
  
if IS_BIASED:
    print("#define _2_OPTIMIZE B00100000 // MULTIPLE_BIASES_PER_LAYER \n")
    print('float biases[] = {')
    for l, (w, b) in enumerate(zip(weights_biases[::2], weights_biases[1::2])):
        print('  ', end='')
        for j in range(0, w.shape[1]):
            print(b[j], end=', ')
        print()
    print('};\n')
else:
    print("#define _2_OPTIMIZE B01000000 // NO_BIAS \n")

print('float weights[] = {', end="")
for l, (w, b) in enumerate(zip(weights_biases[::2], weights_biases[1::2])):
    print()
    for j in range(0, w.shape[1]):
        print('  ', end='')
        for i in range(0, w.shape[0]):
            print(w[i][j], end=', ')
        print()
print('};\n')