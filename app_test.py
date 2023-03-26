# Import librairies
import cv2 # opencv-python
from descriptors import  haralick,haralick_with_mean,bitdesc,glcm

path='test.png'
def main():
    #Read/load image
    img_gray=cv2.imread(path,0) # le chiffre 0 représente retourn image en GrayScale
    print (img_gray.shape)
   
    #print(img_gray)
    print('Haralick features : ')
    haralick_feat=haralick_with_mean(img_gray)
    print(haralick_feat)

    print('Bitdesc features : ')
    bitdesc_feat=bitdesc(img_gray)
    print(bitdesc_feat)   

    print('GLCM features : ')
    glcmz=glcm(img_gray)
    print(glcmz)     
if __name__=="__main__":
    main()