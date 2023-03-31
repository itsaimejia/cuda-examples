import numpy as np

def convolve2D(img, kernel):
    #dimensiones imagen origen y kernel
    img_row, img_col = img.shape
    kernel_row, kernel_col = kernel.shape

    #matriz resultado (imagen) del mismo tamaño de la imagen origen
    img_res = np.zeros(img.shape)

    #definir el tamaño del relleno para cada eje 
    pad_height = int((kernel_row - 1) / 2)
    pad_width = int((kernel_col - 1) / 2)

    #crear una matriz de 0's con el relleno agregado
    padded_img = np.zeros((img_row + (2 * pad_height), img_col + (2 * pad_width)))
    #agregar la imagen origen a la matriz de 0's con 
    padded_img[pad_height:padded_img.shape[0] - pad_height, pad_width:padded_img.shape[1] - pad_width] = img
    
    #recorrer la matriz de la imagen
    for row in range(img_row):
        for col in range(img_col):
            #calcular cada nuevo pixel multiplicando la cada valor del kernel con
            #la seccion de la matriz imagen concidente
            pixel = np.multiply(kernel, padded_img[row:row + kernel_row, col:col + kernel_col])
            #sumar los productos y asignar a la posicion correspondiente de la matriz resultado
            img_res[row, col] = np.sum(pixel)
    
    #se devuelve la matriz resultado verificando que los valores
    #se mantengan en el rango de 0 a 255
    return np.clip(img_res,0,255) 


def gaussianBlur(img):
    '''
    @img: matrix source in grey scale 
    Gaussian blur 3x3
    \t| 1/16 2/16 1/16 |\n
    \t| 2/16 4/16 2/16 |\n
    \t| 1/16 2/16 1/16 |
    \n
    '''
    kernel = np.array([[0.0625, 0.125, 0.0625],
                       [0.125,0.25,0.125],
                       [0.0625, 0.125, 0.0625]])
    result = convolve2D(img, kernel)
    return result



def prewitt(img, axis='x'):
    '''
    @img: matrix source in grey scale 
    @axis: x or y
    Prewitt X
    \t| -1 0 +1 |\n
    \t| -1 0 +1 |\n
    \t| -1 0 +1 |
    \n
    Prewitt Y
    \t| -1 -1 -1 |\n
    \t|  0  0  0 |\n
    \t| +1 +1 +1 |
    \n'''
    if axis == 'x':
        kernel = np.array([[-1,0,1],
                           [-1,0,1],
                           [-1,0,1]])
        result = convolve2D(img, kernel)
        return result
    if axis == 'y':
        kernel = np.array([[-1,-1,-1],
                           [0,0,0],
                           [1,1,1]])
        result = convolve2D(img, kernel)
        return result
   


def sobel(img, axis='x'):
    '''
    @img: matrix source in grey scale 
    @axis: x or y
    Sobel X
    \t| -1 0 1 |\n
    \t| -2 0 2 |\n
    \t| -1 0 1 |
    \n
    Sobel Y
    \t| -1 -2 -1 |\n
    \t|  0  0  0 |\n
    \t| +1 +2 +1 |
    \n'''
    if axis == 'x':
        kernel = np.array([[-1,0,1],
                           [-2,0,2],
                           [-1,0,1]])
        result = convolve2D(img, kernel)
        return result
    elif axis == 'y':
        kernel = np.array([[-1,-2,-1],
                           [0,0,0],
                           [1,2,1]])
        result = convolve2D(img, kernel)
        return result
    
    


def sharpen(img, x=1):
    '''
    @img: matrix source in grey scale 
    @x: level of sharpering
    Sharpen
    \t|  0 -1  0 |\n
    \t| -1 4+x -1 |\n
    \t|  0 -1  0 |
    \n'''
    kernel = np.array([[0,-1,0],
                       [-1,4+x,-1],
                       [0,-1,0]])
    result = convolve2D(img, kernel)
    return result


def edge(img):
    '''
    @img: matrix source in grey scale 
    Edge (outline)
    \t| -1 -1 -1 |\n
    \t| -1 +8 -1 |\n
    \t| -1 -1 -1 |
    \n'''
    kernel = np.array([[-1,-1,-1],
                       [-1,8,-1],
                       [-1,-1,-1]])
    result = convolve2D(img, kernel)
    return result

def average(img):
    '''
    @img: matrix source in grey scale 
    Average (box filter / low blur)
    \t| 1/9 1/9 1/9 |\n
    \t| 1/9 1/9 1/9 |\n
    \t| 1/9 1/9 1/9 |
    \n
    '''
    kernel = np.array([[0.1111111111111111, 0.1111111111111111, 0.1111111111111111],
                      [0.1111111111111111, 0.1111111111111111, 0.1111111111111111],
                      [0.1111111111111111, 0.1111111111111111, 0.1111111111111111]])
    result = convolve2D(img, kernel)
    return result