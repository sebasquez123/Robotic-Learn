import os
import pandas as pd
import numpy as np
import cv2


def recorte(imagen):
    high_shape, width_shape = imagen.shape[:2] #obtiene la altura y ancho de la imagen recortada  
    centro = (high_shape//2 ,width_shape//2) 
    matriz_rotacion = cv2.getRotationMatrix2D(centro, 45, 1.0)
    # Aplicar la rotación
    imagen_rotada = cv2.warpAffine(imagen, matriz_rotacion, (width_shape, high_shape), borderMode=cv2.BORDER_CONSTANT, borderValue=(255, 255, 255))
    # 1) definir el rango de tonos que se desea filtrar en la imagen
    tono_bajo = np.array([20,100,100]) #tonos amarillos con la menor intencidad y brillo bajo de amarillo
    tono_alto = np.array([30,255,255]) #tonos amarillos con la mayor intencidad de luz alta
    tono_rojo_bajo1 = np.array([0, 100, 100])
    tono_rojo_alto1 = np.array([10, 255, 255])
    tono_rojo_bajo2 = np.array([160, 100, 100])
    tono_rojo_alto2 = np.array([180, 255, 255])
    
    # Rango para tonos rosados
    tono_rosado_bajo = np.array([160, 70, 100])   # Rosado más oscuro
    tono_rosado_alto = np.array([180, 150, 255])  

# Rango para tonos de naranja claro (piel)
    tono_naranja_piel_bajo = np.array([5, 50, 120])    
    tono_naranja_piel_alto = np.array([20, 180, 255])  
    

    # 2) convertir la imagen en un espacio de color HSV para poder ser filtrada, y obtener la altura y ancho de la imagen en pixeles
    hsv = cv2.cvtColor(imagen_rotada, cv2.COLOR_BGR2HSV)
    # 3) crea una mascara para filtrar los colores en el rango definido
    
    mask1 = cv2.inRange(hsv, tono_rojo_bajo1, tono_rojo_alto1)
    mask2 = cv2.inRange(hsv, tono_rojo_bajo2, tono_rojo_alto2)
    maskrosado = cv2.inRange(hsv, tono_rosado_bajo, tono_rosado_alto)
    masknaranja = cv2.inRange(hsv, tono_naranja_piel_bajo, tono_naranja_piel_alto)
    maskamarillo = cv2.inRange(hsv, tono_bajo, tono_alto)
    
    
    maskrosadoynaranja = cv2.bitwise_or(maskrosado, masknaranja)
    maskrojo = cv2.bitwise_or(mask1, mask2)
    maskparcial = cv2.bitwise_or(maskamarillo, maskrosadoynaranja)
    
    
    masktotal = cv2.bitwise_or(maskparcial, maskrojo)
    # 4) aplica la mascara a la imagen original
    contours, _ = cv2.findContours(masktotal, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE) #encuentra los contornos de la mascara
    c = max(contours, key=cv2.contourArea) #selecciona el contorno mas grande
    x_shape, y_shape, w_shape, h_shape = cv2.boundingRect(c) #crea el rectangulo mas pequeño alrededor del contorno
    cuadro_amarillo = imagen_rotada[y_shape:y_shape + h_shape+3, x_shape-2:x_shape + w_shape+3] #recorta la imagen original con el rectangulo mas pequeño


    # 5) convertir el cuadro amarillo a formato BGRA (con canal alfa)
    mask_amarillo = masktotal[y_shape:y_shape + h_shape+3, x_shape-2:x_shape + w_shape+3]
    cuadro_amarillo_transparente = cv2.cvtColor(cuadro_amarillo, cv2.COLOR_BGR2BGRA)
    cuadro_amarillo_transparente[:, :, 3] = mask_amarillo
    
    return cuadro_amarillo_transparente
    




def data_augmentation(dir_raw_augmentation,dir_raw):
    

    if not os.path.exists(dir_raw_augmentation):
        os.makedirs(dir_raw_augmentation)
        
##SACAR LA FIGURA 
    imagen_con_mi_cuadrado = cv2.imread(f'{dir_raw}/cuadro_muestra.jpg')
    cuadro_amarillo_rotado = recorte(imagen_con_mi_cuadrado)    
    cv2.imwrite(os.path.join(dir_raw_augmentation, 'cuadro_amarillo.png'), cuadro_amarillo_rotado)
#SACAR EL PLANO

    tono_rojo_bajo1 = np.array([0, 100, 100])
    tono_rojo_alto1 = np.array([10, 255, 255])
    tono_rojo_bajo2 = np.array([160, 100, 100])
    tono_rojo_alto2 = np.array([180, 255, 255])
    #convertir la imagen a HSV
    imagen_del_plano = cv2.imread(f'{dir_raw}/plano_vacio.jpg')
    hsv_plane = cv2.cvtColor(imagen_del_plano, cv2.COLOR_BGR2HSV)
    #crear la mascara con los colores rojos
    mask_plane_1 = cv2.inRange(hsv_plane, tono_rojo_bajo1, tono_rojo_alto1)
    mask_plane_2 = cv2.inRange(hsv_plane, tono_rojo_bajo2, tono_rojo_alto2)
    #generar una mascara general, entre las dos mascaras de rojo
    mask_plane = cv2.bitwise_or(mask_plane_1, mask_plane_2)
    
    #encontrar el contorno de la  imagen mas grande
    contours_plane, _ = cv2.findContours(mask_plane, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    margen_maxima = max(contours_plane, key=cv2.contourArea)

    #crea un rectangulo alrededor del contorno y lo recorta
    x_plane, y_plane, w_plane, h_plane = cv2.boundingRect(margen_maxima)
    plano_vacio_cutted = imagen_del_plano[y_plane:y_plane + h_plane, x_plane:x_plane + w_plane]
    cv2.imwrite(os.path.join(dir_raw_augmentation, 'plano_vacio_cutted.jpg'), plano_vacio_cutted)
    
    
    high_shape, width_shape = cuadro_amarillo_rotado.shape[:2] 
    print(high_shape, width_shape)
    
    

    
    

# SUPERPONER FIGURA EN PLANO

    high_plano, width_plano = plano_vacio_cutted.shape[:2]
    print(high_plano, width_plano)
    high_shape, width_shape = cuadro_amarillo_rotado.shape[:2] 
    
    plano_vacio_cutted = cv2.cvtColor(plano_vacio_cutted, cv2.COLOR_BGR2BGRA)
    plano_vacio_cutted[:, :, 3] = 255  # Canal alfa completamente opaco

    if 27 + high_shape <= plano_vacio_cutted.shape[0] and 27 + width_shape <= plano_vacio_cutted.shape[1]:
        plano_vacio_cutted [27:27+high_shape,27:27+width_shape] = cuadro_amarillo_rotado
        cv2.imwrite(os.path.join(dir_raw_augmentation, 'plano_superpuesto.jpg'), plano_vacio_cutted)

        


    
    