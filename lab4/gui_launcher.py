import tkinter as tk
from tkinter import messagebox, ttk
import subprocess
import sys
import os


class Lab4LauncherGUI:
    def __init__(self, root):
        self.root = root
        self.root.title('4 Задание — CNN')
        self.root.geometry('560x620')
        self.script_dir = os.path.dirname(os.path.abspath(__file__))
        self._build()

    def _build(self):
        tk.Label(
            self.root, text='СМГМО 4 Задание',
            font=('Arial', 20, 'bold'),
        ).pack(pady=12)

        common = tk.LabelFrame(self.root, text='Общие параметры',
                               font=('Arial', 11))
        common.pack(padx=20, pady=6, fill=tk.X)

        tk.Label(common, text='Датасет:', font=('Arial', 10)).grid(
            row=0, column=0, sticky=tk.W, padx=8, pady=6)
        self.dataset = ttk.Combobox(common,
                                    values=['MNIST', 'FashionMNIST'],
                                    state='readonly', width=14)
        self.dataset.set('MNIST')
        self.dataset.grid(row=0, column=1, padx=8, pady=6)

        self._spin(common, 'Train подвыборка:', 1, 0,
                   500, 60000, '10000', attr='tr')
        self._spin(common, 'Test подвыборка:', 1, 2,
                   200, 10000, '2000', attr='te')
        self._spin(common, 'Batch:', 2, 0, 16, 1024, '128', attr='bs')
        self._spin(common, 'N эпох:', 2, 2, 1, 200, '20', attr='ne')
        self._spin(common, 'Patience:', 3, 0, 1, 50, '5', attr='pat')
        self._spin(common, 'Target acc:', 3, 2, 0.5, 1.0, '0.90',
                   inc=0.01, attr='tg')

        ovf = tk.LabelFrame(self.root,
                            text='Демонстрация переобучения (Блок 2)',
                            font=('Arial', 11))
        ovf.pack(padx=20, pady=6, fill=tk.X)

        self._spin(ovf, 'Train (мало):', 0, 0, 50, 5000, '300', attr='ot')
        self._spin(ovf, 'Эпохи:', 0, 2, 10, 1000, '120', attr='oe')

        btns = tk.Frame(self.root)
        btns.pack(pady=12)
        tk.Button(btns, text='Запустить блок 1 (сетка)',
                  width=32, height=2, font=('Arial', 11),
                  command=self.run1).pack(pady=4)
        tk.Button(btns, text='Запустить блок 2 (опт-ры/переобуч.)',
                  width=32, height=2, font=('Arial', 11),
                  command=self.run2).pack(pady=4)
        tk.Button(btns, text='Запустить всё',
                  width=32, height=2, font=('Arial', 11, 'bold'),
                  command=self.run_all).pack(pady=4)

    def _spin(self, parent, label, row, col, frm, to, default,
              attr=None, inc=1):
        tk.Label(parent, text=label, font=('Arial', 10)).grid(
            row=row, column=col, sticky=tk.W, padx=8, pady=6)
        sp = tk.Spinbox(parent, from_=frm, to=to, increment=inc,
                        width=10, font=('Arial', 10))
        sp.delete(0, tk.END)
        sp.insert(0, default)
        sp.grid(row=row, column=col + 1, padx=8, pady=6)
        if attr:
            setattr(self, attr, sp)

    def _env(self):
        env = os.environ.copy()
        try:
            env['SMGMO4_DATASET'] = self.dataset.get()
            env['SMGMO4_TRAIN_SUBSET'] = str(int(self.tr.get()))
            env['SMGMO4_TEST_SUBSET'] = str(int(self.te.get()))
            env['SMGMO4_BATCH_SIZE'] = str(int(self.bs.get()))
            env['SMGMO4_N_EPOCHS'] = str(int(self.ne.get()))
            env['SMGMO4_PATIENCE'] = str(int(self.pat.get()))
            env['SMGMO4_TARGET_ACC'] = str(float(self.tg.get()))
            env['SMGMO4_OVERFIT_TRAIN'] = str(int(self.ot.get()))
            env['SMGMO4_OVERFIT_EPOCHS'] = str(int(self.oe.get()))
        except ValueError:
            messagebox.showerror('Ошибка', 'Проверьте корректность параметров!')
            return None
        return env

    def _run(self, script):
        env = self._env()
        if env is None:
            return
        path = os.path.join(self.script_dir, script)
        try:
            subprocess.Popen([sys.executable, path], env=env,
                             cwd=self.script_dir)
        except Exception as exc:
            messagebox.showerror('Ошибка',
                                 f'Не удалось запустить {script}:\n{exc}')

    def run1(self):
        self._run('block1_main.py')

    def run2(self):
        self._run('block2_main.py')

    def run_all(self):
        self.run1()
        self.run2()


def main():
    root = tk.Tk()
    Lab4LauncherGUI(root)
    root.mainloop()


if __name__ == '__main__':
    main()
