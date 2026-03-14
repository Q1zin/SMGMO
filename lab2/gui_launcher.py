import tkinter as tk
from tkinter import messagebox
import subprocess
import sys
import os


class Lab2LauncherGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("2 Задание")
        self.root.geometry("620x620")
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

        tk.Label(frame, text="Шум:", font=("Arial", 10)).grid(
            row=1, column=0, sticky=tk.W, padx=10, pady=8)

        self.noise_spin = tk.Spinbox(
            frame, from_=0.0, to=3.0, increment=0.1, width=10,
            font=("Arial", 10)
        )
        self.noise_spin.delete(0, tk.END)
        self.noise_spin.insert(0, "0.8")
        self.noise_spin.grid(row=1, column=1, padx=10, pady=8)

        # ── Параметры блока 2 (перцептрон) ─────────────────────────
        p2_frame = tk.LabelFrame(self.root, text="Параметры блока 2",
                                 font=("Arial", 11))
        p2_frame.pack(padx=20, pady=6, fill=tk.X)

        tk.Label(p2_frame, text="Step lr:", font=("Arial", 10)).grid(
            row=0, column=0, sticky=tk.W, padx=10, pady=6)
        self.step_lr_spin = tk.Spinbox(
            p2_frame, from_=0.001, to=5.0, increment=0.001, width=10,
            font=("Arial", 10)
        )
        self.step_lr_spin.delete(0, tk.END)
        self.step_lr_spin.insert(0, "0.03")
        self.step_lr_spin.grid(row=0, column=1, padx=10, pady=6)

        tk.Label(p2_frame, text="Step epochs:", font=("Arial", 10)).grid(
            row=0, column=2, sticky=tk.W, padx=10, pady=6)
        self.step_epochs_spin = tk.Spinbox(
            p2_frame, from_=1, to=50000, increment=1, width=10,
            font=("Arial", 10)
        )
        self.step_epochs_spin.delete(0, tk.END)
        self.step_epochs_spin.insert(0, "1000")
        self.step_epochs_spin.grid(row=0, column=3, padx=10, pady=6)

        tk.Label(p2_frame, text="Sigmoid lr:", font=("Arial", 10)).grid(
            row=1, column=0, sticky=tk.W, padx=10, pady=6)
        self.sigmoid_lr_spin = tk.Spinbox(
            p2_frame, from_=0.001, to=5.0, increment=0.001, width=10,
            font=("Arial", 10)
        )
        self.sigmoid_lr_spin.delete(0, tk.END)
        self.sigmoid_lr_spin.insert(0, "0.03")
        self.sigmoid_lr_spin.grid(row=1, column=1, padx=10, pady=6)

        tk.Label(p2_frame, text="Sigmoid epochs:", font=("Arial", 10)).grid(
            row=1, column=2, sticky=tk.W, padx=10, pady=6)
        self.sigmoid_epochs_spin = tk.Spinbox(
            p2_frame, from_=1, to=50000, increment=1, width=10,
            font=("Arial", 10)
        )
        self.sigmoid_epochs_spin.delete(0, tk.END)
        self.sigmoid_epochs_spin.insert(0, "1000")
        self.sigmoid_epochs_spin.grid(row=1, column=3, padx=10, pady=6)

        # ── Блок 1: по одной кнопке на каждый датасет ──────────────
        b1_frame = tk.LabelFrame(self.root, text="Блок 1: Генерация выборок",
                                  font=("Arial", 11))
        b1_frame.pack(padx=20, pady=6, fill=tk.X)

        DS_NAMES = ['Circles', 'XOR', 'Blobs', 'Spiral']
        for ds in DS_NAMES:
            tk.Button(
                b1_frame, text=ds,
                width=32, height=2, font=("Arial", 11),
                command=lambda d=ds: self.run_block1_ds(d),
            ).pack(pady=4)

        # ── Блок 2 ───────────────────────────────────────────────────
        btn_frame = tk.Frame(self.root)
        btn_frame.pack(pady=8)

        tk.Button(
            btn_frame, text="Блок 2: Элементарный перцептрон",
            width=32, height=3, font=("Arial", 11),
            command=self.run_block2,
        ).pack(pady=5)

    def _run_script(self, script_name, n_str, extra_env=None):
        try:
            n = int(n_str)
            noise = float(self.noise_spin.get())
        except ValueError:
            messagebox.showerror("Ошибка", "Проверьте корректность N и шума!")
            return
        env = os.environ.copy()
        env['SMGMO2_N'] = str(n)
        env['SMGMO2_NOISE'] = str(noise)
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
        try:
            step_lr = float(self.step_lr_spin.get())
            step_epochs = int(self.step_epochs_spin.get())
            sigmoid_lr = float(self.sigmoid_lr_spin.get())
            sigmoid_epochs = int(self.sigmoid_epochs_spin.get())
        except ValueError:
            messagebox.showerror(
                "Ошибка",
                "Проверьте параметры перцептрона: lr и epochs должны быть числами!"
            )
            return

        self._run_script(
            "block2_main.py",
            self.n_spin.get(),
            extra_env={
                'SMGMO2_STEP_LR': str(step_lr),
                'SMGMO2_STEP_EPOCHS': str(step_epochs),
                'SMGMO2_SIGMOID_LR': str(sigmoid_lr),
                'SMGMO2_SIGMOID_EPOCHS': str(sigmoid_epochs),
            },
        )


def main():
    root = tk.Tk()
    Lab2LauncherGUI(root)
    root.mainloop()


if __name__ == "__main__":
    main()
