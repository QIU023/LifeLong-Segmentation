import numpy as np
import os
from scipy.misc import imsave as ImageSaver
from PIL import Image
from matplotlib import pyplot as plt
from  tqdm import tqdm
#from PIL import Image
#from unet import PairwiseDistance



def save_result( prediction, filename ):
    #ImageSaver("results_combine_p/"+filename, prediction)
    ImageSaver("results_combine/"+filename, prediction)
    #image = Image.fromarray(prediction)
    #image.save("fake/"+filename+"_fake.bmp")

def get_file_name( path ):
    return [f for f in os.listdir(path) if f.endswith("png")]

def load_image( image_name ):
    return np.array( Image.open( image_name ) ).astype(float)

def main():
    file_name_list = get_file_name('./results0/')
    for image_name in tqdm(file_name_list):
        image0 = load_image( os.path.join('./results0', image_name) )
        image1 = load_image( os.path.join('./results1', image_name) )
        image2 = load_image( os.path.join('./results2', image_name) )
        #show(image0)
        #show(image1)
        #show(image2)


        image = (image0 + image1 + image2)/3
        image = (image > 128)*255
        #show(image)
        save_result( image, image_name )
def show(image):
    plt.imshow(image)
    plt.show()



if __name__=="__main__":
    main()

