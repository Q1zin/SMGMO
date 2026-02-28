import tkinter as tk
from tkinter import messagebox
import subprocess
import sys
import os

class SMGMOLauncherGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("1 Задание")
        self.root.geometry("400x300")
        
        self.create_widgets()
        
    def create_widgets(self):
        title = tk.Label(self.root, text="СМГМО 1 Задание", font=("Arial", 20, "bold"))
        title.pack(pady=15)
        
        settings_frame = tk.LabelFrame(self.root, text="Размер выборки", font=("Arial", 11))
        settings_frame.pack(padx=20, pady=10, fill=tk.X)
        
        tk.Label(settings_frame, text="Блок 1:", font=("Arial", 10)).grid(row=0, column=0, sticky=tk.W, padx=10, pady=8)
        self.n_block1 = tk.Spinbox(settings_frame, from_=10, to=1000, width=10, font=("Arial", 10))
        self.n_block1.delete(0, tk.END)
        self.n_block1.insert(0, "100")
        self.n_block1.grid(row=0, column=1, padx=10, pady=8)
        
        tk.Label(settings_frame, text="Блок 2:", font=("Arial", 10)).grid(row=1, column=0, sticky=tk.W, padx=10, pady=8)
        self.n_block2 = tk.Spinbox(settings_frame, from_=5, to=100, width=10, font=("Arial", 10))
        self.n_block2.delete(0, tk.END)
        self.n_block2.insert(0, "15")
        self.n_block2.grid(row=1, column=1, padx=10, pady=8)
        
        btn_frame = tk.Frame(self.root)
        btn_frame.pack(pady=15)
        
        tk.Button(btn_frame, text="Блок 1: Генерация выборок", width=30, height=2, command=self.run_block1, font=("Arial", 11)).pack(pady=5)
        tk.Button(btn_frame, text="Блок 2: Полиномиальная регрессия", width=30, height=2, command=self.run_block2, font=("Arial", 11)).pack(pady=5)
        
    def run_script(self, script_name, n_value):
        try:
            n = int(n_value)
            env = os.environ.copy()
            env['SMGMO_N'] = str(n)
            
            subprocess.Popen([sys.executable, script_name], env=env)
        except ValueError:
            messagebox.showerror("Ошибка", "Проверьте корректность размера выборки!")
        except Exception as e:
            messagebox.showerror("Ошибка", f"Не удалось запустить {script_name}:\n{e}")
        
    def run_block1(self):
        n = self.n_block1.get()
        self.run_script("block1_main.py", n)
        
    def run_block2(self):
        n = self.n_block2.get()
        self.run_script("block2_main.py", n)

def main():
    root = tk.Tk()
    app = SMGMOLauncherGUI(root)
    root.mainloop()


if __name__ == "__main__":
    main()
