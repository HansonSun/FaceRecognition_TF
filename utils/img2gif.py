import matplotlib.pyplot as plt
import imageio,os

def convert(inputdir):
    images = []
    filenames=[fn for fn in os.listdir(inputdir) if fn.endswith('.png')]

    filenames=sorted(filenames,key=lambda x:int(x[:-4]) )
    filenames=[os.path.join(inputdir,i) for i in filenames ]

    for filename in filenames:
        images.append(imageio.imread(filename))
    imageio.mimsave('gif.gif', images,duration=0.5)
    
if __name__=="__main__":
    convert("loss_images")