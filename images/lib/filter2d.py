import numpy as np
from numba import jit

list_kernel={
    'gaussian_blur_3x3':[[0.0625, 0.125, 0.0625],
                         [0.125,0.25,0.125],
                         [0.0625, 0.125, 0.0625]],
    'average':[[0.1111111111111111, 0.1111111111111111, 0.1111111111111111],
               [0.1111111111111111, 0.1111111111111111, 0.1111111111111111],
               [0.1111111111111111, 0.1111111111111111, 0.1111111111111111]],
    'prewitt_x':[[-1,0,1],[-1,0,1],[-1,0,1]],
    'prewitt_y':[[-1,-1,-1],[0,0,0],[1,1,1]],
    'sobel_x':[[-1,0,1],[-2,0,2],[-1,0,1]],
    'sobel_y':[[-1,-2,-1],[0,0,0],[1,2,1]],
    'sharpen':[[0,-1,0],[-1,5,-1],[0,-1,0]],
    'edge':[[-1,-1,-1],[-1,8,-1],[-1,-1,-1]]
                
}

@jit
def convolve2D(img, mask):
    img_row, img_col = img.shape
    mask_row, mask_col = mask.shape

    img_res = np.zeros(img.shape)

    pad_height = int((mask_row - 1) / 2)
    pad_width = int((mask_col - 1) / 2)

    padded_img = np.zeros((img_row + (2 * pad_height), img_col + (2 * pad_width)))
    padded_img[pad_height:padded_img.shape[0] - pad_height, pad_width:padded_img.shape[1] - pad_width] = img
    
    for row in range(img_row):
        for col in range(img_col):
            square = np.multiply(mask, padded_img[row:row + mask_row, col:col + mask_col])
            img_res[row, col] = np.sum(square)
    
    return img_res 
@jit
def apply_kernel(img, operator='average'):
    '''
    @img: matrix source in grey scale
    @operator: gaussian_blur_3x3 or average or prewitt_x or prewitt_y sobel_x or sobel_y etc.
    \n
    average (box filter / low blur)
    \t| 1/9 1/9 1/9 |\n
    \t| 1/9 1/9 1/9 |\n
    \t| 1/9 1/9 1/9 |
    \n
    gaussian_blur_3x3
    \t| 1/16 2/16 1/16 |\n
    \t| 2/16 4/16 2/16 |\n
    \t| 1/16 2/16 1/16 |
    \n
    prewitt_x
    \t| -1 0 +1 |\n
    \t| -1 0 +1 |\n
    \t| -1 0 +1 |
    \n
    prewitt_y
    \t| -1 -1 -1 |\n
    \t|  0  0  0 |\n
    \t| +1 +1 +1 |
    \n
    sobel_x
    \t| -1 0 1 |\n
    \t| -2 0 2 |\n
    \t| -1 0 1 |
    \n
    sobel_y
    \t| -1 -2 -1 |\n
    \t|  0  0  0 |\n
    \t| +1 +2 +1 |
    \n
    sharpen
    \t|  0 -1  0 |\n
    \t| -1 +5 -1 |\n
    \t|  0 -1  0 |
    \n
    edge (outline)
    \t| -1 -1 -1 |\n
    \t| -1 +8 -1 |\n
    \t| -1 -1 -1 |
    \n
    '''
    kernel = np.array(list_kernel[operator])
    result = convolve2D(img, kernel)
    return np.clip(result, 0, 255)

   

    # sharpen = np.array([[0,-1,0],
    #                     [-1,5,-1],
    #                     [0,-1,0]])
    # # img_sharpen=convolve2D(img,sharpen)
    # # cv2.imwrite("sharpen.png", img_sharpen)

    # high_edge = np.array([[-1,-1,-1],
    #                       [-1,8,-1],
    #                       [-1,-1,-1]])
    # img_high_egde=convolve2D(img,high_edge)
    # cv2.imwrite("high_edge.png", img_high_egde)

    # laplacian = np.array([[0,-1,0],
    #                       [-1,4,-1],
    #                       [0,-1,0]])
    # # img_laplacian=convolve2D(img,laplacian)
    # # cv2.imwrite("laplacian.png", img_laplacian)