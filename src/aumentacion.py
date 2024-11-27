import os
import pandas as pd
import numpy as np
import cv2
from src.constants import patron_fila,patron_columna
def aplicar_capa(imagen):
    tono_amarillo_bajo = np.array([10,100,100]) #tonos amarillos con la menor intencidad y brillo bajo de amarillo
    tono_amarillo_alto = np.array([40,255,255]) #tonos amarillos con la mayor intencidad de luz alta
    tono_rojo1_bajo = np.array([160, 100, 100]) #tonos rojos con la menor intencidad y brillo bajo de rojo
    tono_rojo1_alto = np.array([180, 255, 255]) #tonos rojos con la mayor intencidad y brillo alto de rojo
    tono_rojo2_bajo = np.array([0, 100, 100]) 
    tono_rojo2_alto = np.array([10, 255, 255])
    imagen_hsv = cv2.cvtColor(imagen, cv2.COLOR_BGR2HSV)
    mask_amarillo = cv2.inRange(imagen_hsv, tono_amarillo_bajo, tono_amarillo_alto)
    mask_rojo1 = cv2.inRange(imagen_hsv, tono_rojo1_bajo, tono_rojo1_alto)
    mask_rojo2 = cv2.inRange(imagen_hsv, tono_rojo2_bajo, tono_rojo2_alto)
    maskrojo = cv2.bitwise_or(mask_rojo1, mask_rojo2)
    return cv2.bitwise_or(mask_amarillo, maskrojo)

def recortar_figura(imagen,rotacion):
    
    #recortar la figura
    
    mask1 = aplicar_capa(imagen)
    contornos, _ = cv2.findContours(mask1, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contorno_maximo = max(contornos, key=cv2.contourArea)
 
    mask2 = np.zeros_like(mask1)
    cv2.drawContours(mask2, [contorno_maximo], -1, 255, -1)
    mascara_negra = cv2.bitwise_and(imagen, imagen, mask=mask2)
    
    x, y, w, h = cv2.boundingRect(contorno_maximo)
    cuadrado_recortado = mascara_negra[y-3:y+h+3, x-3:x+w+3]
    
    #rotar la figura 
    hb, wb = cuadrado_recortado.shape[:2]
    centro = (hb//2, wb//2)
    # M = cv2.getRotationMatrix2D(centro, 180, 1)
    M = cv2.getRotationMatrix2D(centro, rotacion, 1)
    figura_rotada = cv2.warpAffine(cuadrado_recortado, M, (wb , hb), borderValue=(255, 255, 255))
    
    mask3 = aplicar_capa(figura_rotada)
    solo_el_cuadro = cv2.bitwise_and(figura_rotada,figura_rotada,mask=mask3)
    
    return solo_el_cuadro
    


def recortar_plano(imagen):
    return None

def superponer_plano(plano,imagen,angulo,contador,patron_fila,indice_fila,patron_columna,indice_columna):

    xp,p = patron_fila[indice_fila]
    yp = patron_columna[indice_columna]

  ##*por cada 15 grados,menos 3 pixeles en x*##
    if angulo<=90:
        x,y= xp-contador,yp
    if angulo>90 and angulo<=180:
        x,y = xp-p+2,yp+contador 
    if angulo>180 and angulo<=270:
        x,y = xp-p+contador,yp+p
    if angulo>270 and angulo<=365:
        x,y = xp,yp+p-contador
    
    # if angulo<=90:
    #     x,y= 25-contador,33
    # if angulo>90 and angulo<=180:
    #     x,y = 7,33+contador 
    # if angulo>180 and angulo<=270:
    #     x,y = 7+contador,51
    # if angulo>270 and angulo<=365:
    #     x,y = 25,51-contador
        
    w, h = imagen.shape[:2]
    imagen_sintetica = plano.copy()
    roi = imagen_sintetica[y:y+h, x:x+w] #region de interes dentro del plano
    imagen_gris = cv2.cvtColor(imagen, cv2.COLOR_BGR2GRAY)#convertir la imagen a escala de grises
    _,imagen_mask = cv2.threshold(imagen_gris, 1, 255, cv2.THRESH_BINARY)#aplicar umbralizacion
    figura_mask_inv = cv2.bitwise_not(imagen_mask)#invertir la mascara blanco fondo y negro figura
    tablero_bg = cv2.bitwise_and(roi, roi, mask=figura_mask_inv)
    figura_fg = cv2.bitwise_and(imagen, imagen, mask=imagen_mask)
    dst = cv2.add(tablero_bg,figura_fg)
    imagen_sintetica[y:y+h, x:x+w] = dst
    coordenada_y = y
    coordenada_x = x
            
    return imagen_sintetica,coordenada_x,coordenada_y
    

def multiplicar_formato(imagen_con_figura,imagen_con_plano,dir_raw_augmentation):
    
    contador=0
    dataframe=[]
    for indice_columna in range(0,8,1):
        for indice_fila in range(0,8,1):
            contador = 0
            for rotacion in range(0,375,15):
                figura_rotada = recortar_figura(imagen_con_figura,rotacion)  
                imagen_superpuesta,x,y = superponer_plano(imagen_con_plano,figura_rotada,rotacion,contador,patron_fila,indice_fila,patron_columna,indice_columna)
                cv2.imwrite(os.path.join(dir_raw_augmentation, f'sup_x{indice_fila}_y{indice_columna}_{rotacion}G.png'), imagen_superpuesta)
                contador+=3
                if rotacion == 90 or rotacion == 180 or rotacion == 270 or rotacion == 365:
                    contador=3
                if rotacion == 0:
                    dataframe.append([f'sup_x{indice_fila}_y{indice_columna}_{rotacion}G.png',x,y])
                    print(x,y,indice_columna,indice_fila,rotacion)
    return "Generacion de imagenes completado",dataframe

##pipeline principal
def generar_imagenes(dir_raw_augmentation,dir_raw):
    
    imagen_con_figura = cv2.imread(f'{dir_raw}/cuadro_muestra.png')
    imagen_con_plano = cv2.imread(f'{dir_raw}/plano_vacio.png')
    
    if imagen_con_figura is None:
        print('No se encontro la imagen con la figura')
        return
    if imagen_con_plano is None:
        print('No se encontro la imagen del plano')
    
    if not os.path.exists(dir_raw_augmentation):
        os.makedirs(dir_raw_augmentation)
        # plano_recortado = recortar_plano(imagen_de_plano)
    
    mensaje,dataframe = multiplicar_formato(imagen_con_figura,imagen_con_plano,dir_raw_augmentation)
    dataframe = np.array(dataframe)
    print(dataframe)
    print(mensaje)
    # cv2.imwrite(os.path.join(dir_raw_augmentation, 'figura_rotada.png'), figura_rotada)
    # cv2.imwrite(os.path.join(dir_raw_augmentation, 'imagen_superpuesta.png'), imagen_superpuesta)