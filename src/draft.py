from tkinter import *
from threading import Thread

def show_values(x):
    print(x)


def makeslider():
    master = Tk()
    w = Scale(master, from_=-100, to=100, command=show_values)
    w.pack()
    mainloop()

thread = Thread(target = makeslider)
thread.start()

print("Hello")

thread.join()