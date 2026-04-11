"""
GUI-лаунчер для задания 3 — Многослойный перцептрон.
"""

import tkinter as tk
from tkinter import messagebox
import subprocess
import sys
import os


class Lab3LauncherGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("3 Задание — МСП")
        self.root.geometry("560x620")
        self.script_dir = os.path.dirname(os.path.abspath(__file__))
        self._create_widgets()

    # ─────────────────────────────────────────────────────────────
    def _create_widgets(self):
        tk.Label(
            self.root, text="СМГМО 3 Задание",
            font=("Arial", 20, "bold"),
        ).pack(pady=12)

        # ── Регрессия ──────────────────────────────────────────────
        reg_frame = tk.LabelFrame(self.root, text="Регрессия (Блок 1)",
                                  font=("Arial", 11))
        reg_frame.pack(padx=20, pady=6, fill=tk.X)

        self._labeled_spin(reg_frame, "N:", 0, 0, 50, 2000, "280",
                           attr='reg_n')
        self._labeled_spin(reg_frame, "ε₀:", 0, 2, 0.01, 2.0, "0.18",
                           inc=0.01, attr='reg_eps')
        self._labeled_spin(reg_frame, "Epochs:", 1, 0, 50, 10000, "350",
                           attr='reg_epochs')
        self._labeled_spin(reg_frame, "LR:", 1, 2, 0.001, 1.0, "0.02",
                           inc=0.001, attr='reg_lr')

        tk.Button(
            reg_frame, text="Запустить регрессию",
            width=30, height=2, font=("Arial", 11),
            command=self.run_block1,
        ).grid(row=2, column=0, columnspan=4, pady=8)

        # ── Классификация ──────────────────────────────────────────
        cls_frame = tk.LabelFrame(self.root, text="Классификация (Блок 2)",
                                  font=("Arial", 11))
        cls_frame.pack(padx=20, pady=6, fill=tk.X)

        self._labeled_spin(cls_frame, "N:", 0, 0, 50, 5000, "500",
                           attr='cls_n')
        self._labeled_spin(cls_frame, "Шум:", 0, 2, 0.0, 3.0, "0.08",
                           inc=0.01, attr='cls_noise')
        self._labeled_spin(cls_frame, "Epochs:", 1, 0, 50, 10000, "250",
                           attr='cls_epochs')
        self._labeled_spin(cls_frame, "LR:", 1, 2, 0.001, 1.0, "0.03",
                           inc=0.001, attr='cls_lr')

        tk.Button(
            cls_frame, text="Запустить классификацию",
            width=30, height=2, font=("Arial", 11),
            command=self.run_block2,
        ).grid(row=2, column=0, columnspan=4, pady=8)

        # ── Кросс-валидация ────────────────────────────────────────
        cv_frame = tk.LabelFrame(self.root, text="Кросс-валидация",
                                 font=("Arial", 11))
        cv_frame.pack(padx=20, pady=6, fill=tk.X)

        self._labeled_spin(cv_frame, "K folds:", 0, 0, 2, 20, "5",
                           attr='cv_k')

        # ── Оба блока ─────────────────────────────────────────────
        btn_frame = tk.Frame(self.root)
        btn_frame.pack(pady=10)

        tk.Button(
            btn_frame, text="Запустить всё",
            width=30, height=2, font=("Arial", 11, "bold"),
            command=self.run_all,
        ).pack(pady=5)

    # ─────────────────────────────────────────────────────────────
    def _labeled_spin(self, parent, label, row, col, from_, to, default,
                      inc=1, attr=None):
        tk.Label(parent, text=label, font=("Arial", 10)).grid(
            row=row, column=col, sticky=tk.W, padx=8, pady=6)
        spin = tk.Spinbox(parent, from_=from_, to=to, increment=inc,
                          width=10, font=("Arial", 10))
        spin.delete(0, tk.END)
        spin.insert(0, default)
        spin.grid(row=row, column=col + 1, padx=8, pady=6)
        if attr:
            setattr(self, attr, spin)

    # ─────────────────────────────────────────────────────────────
    def _build_env(self, block):
        env = os.environ.copy()
        try:
            if block in ('reg', 'all'):
                env['SMGMO3_REG_N'] = str(int(self.reg_n.get()))
                env['SMGMO3_REG_EPS0'] = str(float(self.reg_eps.get()))
                env['SMGMO3_REG_EPOCHS'] = str(int(self.reg_epochs.get()))
                env['SMGMO3_REG_LR'] = str(float(self.reg_lr.get()))
            if block in ('cls', 'all'):
                env['SMGMO3_CLS_N'] = str(int(self.cls_n.get()))
                env['SMGMO3_CLS_NOISE'] = str(float(self.cls_noise.get()))
                env['SMGMO3_CLS_EPOCHS'] = str(int(self.cls_epochs.get()))
                env['SMGMO3_CLS_LR'] = str(float(self.cls_lr.get()))
                env['SMGMO3_CV_K'] = str(int(self.cv_k.get()))
        except ValueError:
            messagebox.showerror("Ошибка", "Проверьте корректность параметров!")
            return None
        return env

    def _run_script(self, script, block):
        env = self._build_env(block)
        if env is None:
            return
        path = os.path.join(self.script_dir, script)
        try:
            subprocess.Popen([sys.executable, path], env=env,
                             cwd=self.script_dir)
        except Exception as exc:
            messagebox.showerror("Ошибка",
                                 f"Не удалось запустить {script}:\n{exc}")

    # ─────────────────────────────────────────────────────────────
    def run_block1(self):
        self._run_script('block1_main.py', 'reg')

    def run_block2(self):
        self._run_script('block2_main.py', 'cls')

    def run_all(self):
        self._run_script('block1_main.py', 'all')
        self._run_script('block2_main.py', 'all')


def main():
    root = tk.Tk()
    Lab3LauncherGUI(root)
    root.mainloop()


if __name__ == '__main__':
    main()
