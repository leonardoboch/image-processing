import numpy as np
import cv2

IMAGE_A = "./pacote-trabalho3/Wind Waker GC.bmp"
IMAGE_B = "./pacote-trabalho3/GT2.BMP"
# é uma versão do filtro da media utilizado no trabalho 2
# mais caprichado, estruturado
def box_blur(img, kernel_width, kernel_height):
    altura,largura,canais = img.shape


    direita =  (kernel_width // 2) * (-1)
    esquerda = (kernel_width // 2) + 1

    g_linhas = np.zeros(img.shape, np.float32)


    #ignorando as bordas
    for y in range(0, altura - kernel_height + 1):
        for x in range(0, largura - kernel_width + 1):
            for c in range(0, canais):
                soma = 0.0
                # contador de somas para fazer a media das linhas
                count = 0
                for i in range(direita, esquerda):
                    if 0 <= (x + i) < largura:
                        soma += img[y, x + i, c]
                        count += 1
                if count > 0:
                    g_linhas[y,x,c] = soma / count
                else:
                    g_linhas[y, x, c] = 0

    ## agora para as linhas
    g = np.zeros(img.shape, np.float32)
    cima = (kernel_height // 2) * (-1)
    baixo = (kernel_height // 2) + 1

    # ignorando as bordas
    for y in range(0, altura - kernel_height + 1):
        for x in range(0, largura - kernel_width + 1):
            for c in range(0, canais):
                soma = 0.0
                # contador de somas para fazer a media das linhas
                count = 0
                for i in range(cima, baixo):
                    if 0 <= (y + i) < altura:
                        soma += g_linhas[y, x + i, c]
                        count += 1
                if count > 0:
                    g[y, x, c] = float(soma / count)
                else:
                    g[y, x, c] = 0

    return g


def bright_pass(mask, img):
    altura, largura = mask.shape

    for y in range(0, altura):
        for x in range(0, largura):
            if mask[y,x] == 0:
                img[y,x] = (img[y,x] * 0)

    return img

def teste(img):
    # faz uma escala de luminosidade dependendo da cor do pixel.
    luminosidade = 0.2126 * img[:, :, 2] + 0.7152 * img[:, :, 1] + 0.0722 * img[:, :, 0]
    threshold = 0.5
    bright_pass = np.where(luminosidade > threshold, luminosidade, 0)
    return bright_pass


#esse aqui veio do trabalho 01
def binariza(img, threshold):
    ''' Binarização simples por limiarização.

    Parâmetros:
        img: imagem de entrada. Se tiver mais que 1 canal, binariza cada canal independentemente.
        threshold: limiar.

    Valor de retorno: versão binarizada da img_in.'''
    img[:] = np.where(img > threshold, 1, 0)

    return img


def main():
    print("main")
    img_a = cv2.imread(IMAGE_A)
    img_b = cv2.imread(IMAGE_B)
    img_a_grayscale = cv2.imread(IMAGE_A, cv2.IMREAD_GRAYSCALE)
    img_b_grayscale = cv2.imread(IMAGE_B, cv2.IMREAD_GRAYSCALE)
    img_b = img_b/255
    img_a = img_a/255
    img_b_grayscale = img_b_grayscale/255
    img_a_grayscale = img_a_grayscale/255
    mask_a = box_blur(bright_pass(binariza(img_a_grayscale, 0.47), img_a), 10,10)

    for i in range (0,10):
        mask_a = box_blur(mask_a, 10, 10)

    # mask_b = box_blur(bright_pass(binariza(img_b_grayscale, 0.47), img_b), 10,10)

    #g_final = img_a + mask_a
    cv2.imshow("final", mask_a)





    #box_blur(img_b, 3)

    #cv2.imshow("image_b", img_b)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
