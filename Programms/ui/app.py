import os
import sys
import tkinter as tk
from tkinter import messagebox
import customtkinter as ctk
import os
import threading
import queue
import pandas as pd
import subprocess
import platform
from pathlib import Path
from csv_resume import csv_resume_file
from process import mm
from utils import logs_path, results_path, config_path, training_path, inference_path


class Logger(object):
    def __init__(self, type_):
        stream = type_.name.replace('<','').replace('>','')
        if stream == "stdout" :
            self.terminal = sys.stdout
        else :
            self.terminal = sys.stderr
        self.log = open(os.path.join(logs_path, type_.name.replace('<','').replace('>','') + ".log"), "w")
   
    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)  

    def flush(self):
        self.terminal.flush()
        self.log.flush()    


class MyApp:
    def __init__(self, root):
        self.CVO = ["All conditions", "Histological & clinical values", "Histological values & DFG at 3 months", "Histological values & Age", "No clinical values & No DFG", "Clinical values only"]
        self.root = root
        self.root.title("Skinet - transplantML")
        self.root.geometry("800x500")
        self.root.resizable(False, False)
        self.root.configure(bg='#F0F4F8')
        self.mode = "Training"
        self.main_application_ui()

    def load_main_ui(self):
        self.mode = self.mode_.get()

    def main_application_ui(self):
        # Clear the window
        for widget in self.root.winfo_children():
            widget.destroy()

        # Configure the grid layout
        self.root.columnconfigure(0, weight=1)
        self.root.columnconfigure(1, weight=1)
        self.root.columnconfigure(2, weight=1)
        self.root.rowconfigure(0, weight=2)
        self.root.rowconfigure(1, weight=2)
        self.root.rowconfigure(2, weight=2)
        self.root.rowconfigure(3, weight=2)
        self.root.rowconfigure(4, weight=2)

        # Create and place widgets
        self.mode_ = ctk.StringVar(self.root, "Training")
        self.training = ctk.CTkRadioButton(self.root, text="Training", font=('Helvetica', 20), variable=self.mode_, value="Training", command=self.load_main_ui)
        self.inference = ctk.CTkRadioButton(self.root, text="Inference", font=('Helvetica', 20), variable=self.mode_, value="Inference", command=self.load_main_ui)
        self.biopsy_entry = ctk.CTkEntry(self.root, text_color='black', fg_color='#FFFFFF', border_color='#CBD2D9')
        self.biopsy_label = tk.Label(self.root, text="Exclude biopsy #", font=('Helvetica', 15), bg='#F0F4F8', fg='#333333')
        self.biopsy_button = ctk.CTkButton(self.root, text="Exclude", fg_color='#696969', hover_color='#A9A9A9', command=self.exclude)
        self.run_button = ctk.CTkButton(self.root, text="Run", bg_color="#F0F4F8", fg_color="#006400", hover_color="#008000", command=self.start_run_thread)
        self.quit_button = ctk.CTkButton(self.root, text="Quit", fg_color='#696969', hover_color='#A9A9A9', command=self.quit_application)
        self.tab_button = ctk.CTkButton(self.root, text="Results", fg_color='#696969', hover_color='#A9A9A9', command=self.open_tab)

        self.combo_box = ctk.CTkOptionMenu(self.root, values=self.CVO, button_color='#696969', fg_color='#696969', button_hover_color='#A9A9A9')

        self.check_box = ctk.CTkFrame(self.root, fg_color='#F0F4F8', bg_color='#F0F4F8')

        self.check_all_years = ctk.CTkCheckBox(self.check_box, text="All years", font=('Helvetica', 15), text_color='#333333', checkmark_color='#333333', hover_color='#D9E2EC', fg_color='#85A8AA', border_color='#333333', onvalue='All years', offvalue=0)
        self.check_1_years   = ctk.CTkCheckBox(self.check_box, text="1 year"   , font=('Helvetica', 15), text_color='#333333', checkmark_color='#333333', hover_color='#D9E2EC', fg_color='#85A8AA', border_color='#333333', onvalue='1 an Clair CKDEPI en ml/min/1.73m2', offvalue=0)
        self.check_3_years   = ctk.CTkCheckBox(self.check_box, text="3 years"  , font=('Helvetica', 15), text_color='#333333', checkmark_color='#333333', hover_color='#D9E2EC', fg_color='#85A8AA', border_color='#333333', onvalue='3 an Clair CKDEPI en ml/min/1.73m2', offvalue=0)
        self.check_5_years   = ctk.CTkCheckBox(self.check_box, text="5 years"  , font=('Helvetica', 15), text_color='#333333', checkmark_color='#333333', hover_color='#D9E2EC', fg_color='#85A8AA', border_color='#333333', onvalue='5 an Clair CKDEPI en ml/min/1.73m2', offvalue=0)
        self.check_7_years   = ctk.CTkCheckBox(self.check_box, text="7 years"  , font=('Helvetica', 15), text_color='#333333', checkmark_color='#333333', hover_color='#D9E2EC', fg_color='#85A8AA', border_color='#333333', onvalue='7 an Clair CKDEPI en ml/min/1.73m2', offvalue=0)
        
        self.check_box.columnconfigure(0, weight=1)
        self.check_box.columnconfigure(1, weight=1)
        self.check_box.columnconfigure(2, weight=1)
        self.check_box.columnconfigure(3, weight=1)
        self.check_box.columnconfigure(4, weight=1)

        self.training.grid(column=0, row=0, columnspan=1)
        self.inference.grid(column=2, row=0, columnspan=1)
        self.biopsy_label.grid(row=1, column=0)
        self.biopsy_entry.grid(row=1, column=1)
        self.biopsy_button.grid(row=1, column=2)
        self.check_box.grid(column=0, row=2, sticky=tk.EW, columnspan=3)
        self.check_all_years.grid(row=0, column=0)
        self.check_1_years.grid(row=0, column=1)
        self.check_3_years.grid(row=0, column=2)
        self.check_5_years.grid(row=0, column=3)
        self.check_7_years.grid(row=0, column=4)
        self.combo_box.grid(row=3, columnspan=3)
        self.quit_button.grid(column=0, row=4)
        self.run_button.grid(column=1, row=4)
        self.tab_button.grid(column=2, row=4)

        self.progress = ctk.CTkProgressBar(
            self.root, orientation="horizontal", mode="determinate", fg_color='#85A8AA', progress_color='#006400')
        self.progress.grid(column=0, row=5, columnspan=3,
                           padx=20, pady=10, sticky=tk.EW)
        self.progress_label = tk.Label(
            self.root, text="", bg='#F0F4F8', fg='#333333')
        self.progress_label.grid(column=0, row=6, columnspan=3, sticky=tk.NSEW)
        self.progress.grid_remove()
        self.progress_label.grid_remove()
        self.queue = queue.Queue()

    def start_run_thread(self):
        if os.path.exists(results_path / 'models'):
            if not messagebox.askyesno("Confirmation", "The results folder already exists. Do you want to overwrite it ?"):
                return
        if os.path.exists(results_path / 'No clinical value'):
            if not messagebox.askyesno("Confirmation", "The results folder already exists. Do you want to overwrite it ?"):
                return
        self.thread = threading.Thread(target=self.run)
        self.thread.daemon = True
        self.thread.start()

    def stop_thread(self):
        if messagebox.askokcancel("Stop", "Are you sure you want to stop the current process ?"):
            self.run_button.configure(text="Run", fg_color='#006400', hover_color='#008000', command=self.start_run_thread)
            self.hide_progress()

    def run(self):
        
        print(f"INFO : Running mode \"{self.mode}\"")

        cvo = self.combo_box.get()
        if self.check_1_years.get() == 0 and self.check_3_years.get() == 0 and self.check_5_years.get() == 0 and self.check_7_years.get() == 0 and self.check_all_years.get() == 0:
            messagebox.showerror("Warning", "Please choose at least one simulation !")
        else:
            self.run_button.configure(text="Stop", fg_color='#FF0000', bg_color='#F0F4F8', hover_color='#FF7F7F', command=self.stop_thread)
            years = []
            if self.check_1_years.get() == '1 an Clair CKDEPI en ml/min/1.73m2':
                years.append(self.check_1_years.get())
            if self.check_3_years.get() == '3 an Clair CKDEPI en ml/min/1.73m2':
                years.append(self.check_3_years.get())
            if self.check_5_years.get() == '5 an Clair CKDEPI en ml/min/1.73m2':
                years.append(self.check_5_years.get())
            if self.check_7_years.get() == '7 an Clair CKDEPI en ml/min/1.73m2':
                years.append(self.check_7_years.get())
            if self.check_all_years.get() == "All years":
                years.append('1 an Clair CKDEPI en ml/min/1.73m2')
                years.append('3 an Clair CKDEPI en ml/min/1.73m2')
                years.append('5 an Clair CKDEPI en ml/min/1.73m2')
                years.append('7 an Clair CKDEPI en ml/min/1.73m2')
            
            if not os.path.exists(results_path):
                os.mkdir(results_path)

            if self.mode == self.mode :

                self.queue.put(("start", len(years)))

                if not os.path.exists(results_path / self.mode):
                    os.mkdir(results_path / self.mode)
                for folder in years:
                    if not os.path.exists(results_path / self.mode / f'{folder[0]}_years'):
                        os.mkdir(results_path / self.mode / f'{folder[0]}_years')
                    if not os.path.exists(results_path / self.mode / f'{folder[0]}_years'):
                        os.mkdir(results_path / self.mode / f'{folder[0]}_years' / 'models')

                for _, choice in enumerate(years):

                    print(f"INFO : Running year \"{choice}\"")

                    if cvo == 'All conditions':
                        for cvo_ in self.CVO[1:] :
                            mm(self.mode, choice, cvo_)
                    else :
                        mm(self.mode, choice, cvo)
        print("INFO : Run completed")  
        self.run_button.configure(text="Run", fg_color='#006400', hover_color='#008000', bg_color='#F0F4F8', command=self.start_run_thread)

    def process_queue(self):
        try:
            while True:
                msg = self.queue.get_nowait()
                if msg[0] == "error":
                    messagebox.showwarning("Warning", msg[1])
                elif msg[0] == "start":
                    self.progress.grid()
                    self.progress_label.grid()
                    self.progress_label.config(text="Running simulations, please wait...")
                    self.progress.set(0)
                elif msg[0] == "progress":
                    self.progress.set(msg[1])
                elif msg[0] == "complete":
                    self.progress_label.config(text="Simulations completed")
                    self.root.after(2000, self.hide_progress)
        except queue.Empty:
            self.root.after(100, self.process_queue)

    def hide_progress(self):
        self.progress.grid_remove()
        self.progress_label.grid_remove()

    def open_tab(self):
        current_dir = Path(__file__).resolve().parent.parent.parent
        tab_file = current_dir / "result" / "comparaison_des_modeles_predictions.txt"
        if self.combo_box.get() == 'No Clinical Values':
            csv_resume_file(results_path, 'No clinical value')
        if platform.system() == "Linux":
            subprocess.run(['xdg-open', results_path])
        if platform.system() == "Windows":
            os.startfile(results_path)
        if platform.system() == "Darwin":
            subprocess.run(['open', results_path])

    def quit_application(self):
        if messagebox.askokcancel("Quit", "Are you sure you want to quit ?"):
            self.root.destroy()

    def verifier_donnee(self, donnee, df, column_name):
        return donnee in df[column_name].values

    def exclude(self):
        if not os.path.exists(config_path):
            os.mkdir(config_path)
        if self.biopsy_entry.get() == "":
            messagebox.showwarning("Warning", "Please enter a biopsy number !")
        else:
            if self.mode == "Training mode":
                config_file = config_path / self.mode / 'excluded.txt'
                data_path = training_path
            else :
                config_file = config_path / "Inference" / 'excluded.txt'
                data_path = inference_path
            biopsy_number = self.biopsy_entry.get().replace('\n', '').upper()
            if os.path.exists(config_file):
                with open(config_file, 'r') as read_file:
                    exclude_entries = read_file.readlines()
                    if any(biopsy_number == row.replace('\n', '').upper() for row in exclude_entries):
                        response = messagebox.askyesnocancel("Warning", "Biopsy number already excluded. Do you want to remove from excluded list ?")
                        if response is None:  # Cancel
                            self.biopsy_entry.delete(0, tk.END)
                            return
                        elif response:  # Yes, delete
                            with open(config_file, 'w') as write_file:
                                for row in exclude_entries:
                                    if row.replace('\n', '').upper() != biopsy_number:
                                        write_file.write(row)
                            messagebox.showinfo("Info", "Biopsy number has been removed from excluded list")
                        self.biopsy_entry.delete(0, tk.END)
                        return
            df = pd.read_excel(data_path)
            if self.verifier_donnee(biopsy_number, df, 'N° Biopsie'):
                with open(config_file, 'a') as file:
                    file.write(biopsy_number + '\n')
                    messagebox.showinfo("Info", "Biopsy number has been excluded")
                    self.biopsy_entry.delete(0, tk.END)
                    return
            else:
                messagebox.showwarning("Warning", "Biopsy number does not exist")
                self.biopsy_entry.delete(0, tk.END)


def tkapp():
    if not os.path.exists(logs_path) :
        os.mkdir(logs_path)
    sys.stdout = Logger(sys.stdout)
    sys.stderr = Logger(sys.stderr)
    root = tk.Tk()
    MyApp(root)
    root.mainloop()
