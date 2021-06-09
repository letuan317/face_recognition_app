from tkinter import *
from PIL import ImageTk, Image
root = Tk()

canv = Canvas(root, width=300, height=300, bg='white')
canv.grid(row=2, column=3)

img = ImageTk.PhotoImage(Image.open("logo.jpg"))  # PIL solution
canv.create_image(20, 20, anchor=NW, image=img)

mainloop()