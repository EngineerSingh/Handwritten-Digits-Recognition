import matplotlib.pyplot as plt
import numpy as np
import PIL
from PIL import Image
from tkinter import *

def notepad():
    class mouse:
        def __init__(self):
            self.xx,self.yy = [],[]
        def motion(self,event):
            #print("Mouse position: (%s %s)" % (event.x, event.y))
            w.create_oval(event.x, event.y,event.x+14, event.y+14,fill = 'black',width=10)
            self.xx.append(event.x)
            self.yy.append(event.y)
        def getp(self):
            return self.xx,self.yy
        def resetp(self):
            self.xx,self.yy = [],[]
    c = mouse()
    win = Tk()
    w = Canvas(win, width=300, height=300, bg="white")
    w.config(bg='white')
    w.bind('<B1-Motion>',c.motion)
    w.pack()

    mainloop()

    x,y = c.getp()
    #y = np.array(y)
    #y = max(y)-y
    #plt.xticks([i for i in range(1000,1,-25)])
    #plt.yticks([i for i in range(1000,1,-25)])
    #plt.ylim(300,1)
    #plt.xlim(1,300)
    #plt.plot(x,y,linewidth=10)
    #plt.show()



    # make an agg figure
    fig, ax = plt.subplots()
    plt.ylim(300,1)
    plt.xlim(1,300)
    ax.plot(x,y,linewidth=10)
    fig.canvas.draw()
    # grab the pixel buffer and dump it into a numpy array
    X = np.array(fig.canvas.renderer._renderer)
    # now display the array X as an Axes in a new figure
    fig2 = plt.figure()
    ax2 = fig2.add_subplot(111, frameon=False)
    #ax2.imshow(X)
    #plt.axis('off')
    #plt.grid(False)
    #plt.show()
    from PIL import Image
    im = Image.fromarray(X)
    im = im.convert('L')
    #plt.imshow(im)
    #plt.show()
    arr = np.asarray(im)
    #print(arr.shape)
    arr2 = arr[40:-40,60:-50]
    #arr2.shape
    im = Image.fromarray(arr2)
    plt.imshow(im)
    plt.axis('off')
    plt.grid(False)
    plt.show()
    im = Image.fromarray(arr2)
    im.save("imm.jpg")