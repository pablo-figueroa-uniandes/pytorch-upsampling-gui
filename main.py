import tkinter as tk

from gui import App

if __name__ == "__main__":
    root = tk.Tk()
    root.title("Upsampling Comparison")
    App(root).pack(fill="both", expand=True)
    root.mainloop()
