import tkinter as tk
from tkinter import messagebox
import subprocess
import sys
import os


class Lab2LauncherGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("2 Задание")
        self.root.geometry("400x390")
        self.script_dir = os.path.dirname(os.path.abspath(__file__))
        self._create_widgets()

    def _create_widgets(self):
        tk.Label(
            self.root, text="СМГМО 2 Задание",
            font=("Arial", 20, "bold"),
        ).pack(pady=15)

        frame = tk.LabelFrame(self.root, text="Размер выборки",
                               font=("Arial", 11))
        frame.pack(padx=20, pady=5, fill=tk.X)

        tk.Label(frame, text="N:", font=("Arial", 10)).grid(
            row=0, column=0, sticky=tk.W, padx=10, pady=8)

        self.n_spin = tk.Spinbox(frame, from_=50, to=2000, width=10,
                                  font=("Arial", 10))
        self.n_spin.delete(0, tk.END)
        self.n_spin.insert(0, "200")
        self.n_spin.grid(row=0, column=1, padx=10, pady=8)

        # ── Блок 1: по одной кнопке на каждый датасет ──────────────
        b1_frame = tk.LabelFrame(self.root, text="Блок 1: Генерация выборок",
                                  font=("Arial", 11))
        b1_frame.pack(padx=20, pady=6, fill=tk.X)

        DS_NAMES = ['Circles', 'XOR', 'Blobs', 'Spiral']
        for ds in DS_NAMES:
            tk.Button(
                b1_frame, text=ds,
                width=28, height=1, font=("Arial", 10),
                command=lambda d=ds: self.run_block1_ds(d),
            ).pack(pady=3)

        # ── Блок 2 ───────────────────────────────────────────────────
        btn_frame = tk.Frame(self.root)
        btn_frame.pack(pady=8)

        tk.Button(
            btn_frame, text="Блок 2: Элементарный перцептрон",
            width=30, height=2, font=("Arial", 11),
            command=self.run_block2,
        ).pack(pady=5)

    def _run_script(self, script_name, n_str, extra_env=None):
        try:
            n = int(n_str)
        except ValueError:
            messagebox.showerror("Ошибка", "Проверьте корректность N!")
            return
        env = os.environ.copy()
        env['SMGMO2_N'] = str(n)
        if extra_env:
            env.update(extra_env)
        script_path = os.path.join(self.script_dir, script_name)
        try:
            subprocess.Popen(
                [sys.executable, script_path],
                env=env,
                cwd=self.script_dir,
            )
        except Exception as exc:
            messagebox.showerror(
                "Ошибка", f"Не удалось запустить {script_name}:\n{exc}")

    def run_block1_ds(self, ds_name):
        self._run_script("block1_main.py", self.n_spin.get(),
                         extra_env={'SMGMO2_DS': ds_name})

    def run_block2(self):
        self._run_script("block2_main.py", self.n_spin.get())


def main():
    root = tk.Tk()
    Lab2LauncherGUI(root)
    root.mainloop()


if __name__ == "__main__":
    main()
