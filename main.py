import os
from src.preprocess import procesar_imagen, crear_dataframe
from src.modelo import preentrenamiento, entrenamiento_final
from src.test import test_model
from src.data_augmentation import data_augmentation
from src.aumentacion import generar_imagenes
import numpy as np
import pandas as pd
dir_processed = 'data/processed'
dir_raw = 'data/raw'
dir_labels = 'data/labels/images.csv'
dir_test = 'data/test'
dir_raw_augmentation = 'data/raw_augmentation'
# if not os.path.exists(dir_processed):
#     os.makedirs(dir_processed)

# procesar_imagen(dir_raw, dir_processed)

# df= crear_dataframe(dir_processed,dir_labels)


# print(df)
# # entrenamiento parcial para encontrar la mejor profundidad del arbol
# preentrenamiento(df) #comentar en produccion
# # entrenamiento final con la mejor profundidad
# entrenamiento_final(20,df) 



# y = test_model(dir_test)
# print(y)
generar_imagenes(dir_raw_augmentation,dir_raw) #comentar en produccion
# data_augmentation(dir_raw_augmentation,dir_raw) #comentar en produccion

