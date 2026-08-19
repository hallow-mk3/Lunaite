"""
Lunaite AI 10B — Native Desktop Trainer (Zero Web, Pure Desktop GUI)
===================================================================
A native desktop window with a file picker and a single "Train Model" button.

Usage:
    python gui.py
"""

import os
import sys
import threading
import customtkinter as ctk
from tkinter import filedialog, messagebox

# Set appearance
ctk.set_appearance_mode("dark")
ctk.set_default_color_theme("blue")

class LunaiteDesktopApp(ctk.CTk):
    def __init__(self):
        super().__init__()

        self.title("Lunaite AI 10B — Desktop Trainer")
        self.geometry("640x620")
        self.resizable(False, False)

        # Header
        self.header_frame = ctk.CTkFrame(self, fg_color="#0e1117", corner_radius=10)
        self.header_frame.pack(fill="x", padx=16, pady=(16, 10))

        self.title_label = ctk.CTkLabel(
            self.header_frame, 
            text="🌙 Lunaite AI 10B Trainer", 
            font=ctk.CTkFont(size=20, weight="bold")
        )
        self.title_label.pack(pady=(10, 2))

        self.sub_label = ctk.CTkLabel(
            self.header_frame, 
            text="Standalone Model Fine-Tuning Engine — Created by Swasthik Shetty", 
            font=ctk.CTkFont(size=12),
            text_color="#94a3b8"
        )
        self.sub_label.pack(pady=(0, 10))

        # Dataset Section
        self.dataset_frame = ctk.CTkFrame(self, corner_radius=10)
        self.dataset_frame.pack(fill="x", padx=16, pady=8)

        self.ds_label = ctk.CTkLabel(self.dataset_frame, text="Dataset File (.md, .jsonl, .csv, .json):", font=ctk.CTkFont(size=13, weight="bold"))
        self.ds_label.pack(anchor="w", padx=14, pady=(10, 4))

        self.path_frame = ctk.CTkFrame(self.dataset_frame, fg_color="transparent")
        self.path_frame.pack(fill="x", padx=14, pady=(0, 12))

        self.path_entry = ctk.CTkEntry(self.path_frame, placeholder_text="Select dataset.md or file...")
        self.path_entry.insert(0, os.path.abspath("dataset.md") if os.path.exists("dataset.md") else "dataset.md")
        self.path_entry.pack(side="left", fill="x", expand=True, padx=(0, 8))

        self.browse_btn = ctk.CTkButton(self.path_frame, text="Browse", width=80, command=self.browse_file)
        self.browse_btn.pack(side="right")

        # Settings
        self.settings_frame = ctk.CTkFrame(self, corner_radius=10)
        self.settings_frame.pack(fill="x", padx=16, pady=8)

        # Model Selector
        self.m_label = ctk.CTkLabel(self.settings_frame, text="Foundation Base Model:", font=ctk.CTkFont(size=12))
        self.m_label.grid(row=0, column=0, padx=14, pady=8, sticky="w")
        self.model_combo = ctk.CTkComboBox(
            self.settings_frame, 
            values=["Qwen/Qwen2.5-1.5B (Fast)", "Qwen/Qwen2.5-7B (10B Scale)", "meta-llama/Llama-3.1-8B"],
            width=260
        )
        self.model_combo.set("Qwen/Qwen2.5-1.5B (Fast)")
        self.model_combo.grid(row=0, column=1, padx=14, pady=8, sticky="e")

        # Epochs
        self.ep_label = ctk.CTkLabel(self.settings_frame, text="Epochs:", font=ctk.CTkFont(size=12))
        self.ep_label.grid(row=1, column=0, padx=14, pady=8, sticky="w")
        self.epochs_slider = ctk.CTkSlider(self.settings_frame, from_=1, to=10, number_of_steps=9, width=200)
        self.epochs_slider.set(3)
        self.epochs_slider.grid(row=1, column=1, padx=14, pady=8, sticky="e")

        # Progress Bar & Status
        self.status_label = ctk.CTkLabel(self, text="Ready to train", font=ctk.CTkFont(size=12), text_color="#94a3b8")
        self.status_label.pack(pady=(12, 4))

        self.progress_bar = ctk.CTkProgressBar(self, width=580, height=8)
        self.progress_bar.set(0)
        self.progress_bar.pack(pady=4)

        # Log output
        self.log_text = ctk.CTkTextbox(self, width=580, height=140, font=ctk.CTkFont(family="Consolas", size=11))
        self.log_text.pack(pady=10)
        self.log_text.insert("end", "[*] Desktop Trainer Ready. Select your dataset and click 'TRAIN MODEL'.\n")

        # Big Train Button
        self.train_btn = ctk.CTkButton(
            self, 
            text="⚡ TRAIN MODEL", 
            font=ctk.CTkFont(size=16, weight="bold"),
            height=48,
            fg_color="#00d2ff",
            text_color="#000000",
            hover_color="#00b4db",
            command=self.start_training
        )
        self.train_btn.pack(fill="x", padx=16, pady=(4, 16))

        self.is_training = False

    def browse_file(self):
        f = filedialog.askopenfilename(filetypes=[("Datasets", "*.md *.jsonl *.csv *.json *.txt")])
        if f:
            self.path_entry.delete(0, "end")
            self.path_entry.insert(0, f)

    def log(self, text):
        self.log_text.insert("end", f"{text}\n")
        self.log_text.see("end")

    def start_training(self):
        if self.is_training:
            return
        
        path = self.path_entry.get().strip()
        if not os.path.exists(path):
            messagebox.showerror("Error", f"Dataset file not found: {path}")
            return

        self.is_training = True
        self.train_btn.configure(state="disabled", text="⏳ TRAINING IN PROGRESS...")
        self.status_label.configure(text="Training started...", text_color="#00d2ff")
        self.progress_bar.set(0.1)

        t = threading.Thread(target=self.run_train_thread, args=(path,), daemon=True)
        t.start()

    def run_train_thread(self, path):
        try:
            from train_lunaite_lora import train_lunaite
            model_sel = self.model_combo.get().split()[0]
            epochs = int(self.epochs_slider.get())

            self.log(f"[*] Ingesting dataset: {path}")
            self.log(f"[*] Training on base model: {model_sel} ({epochs} epochs)...")

            self.progress_bar.set(0.3)

            meta = train_lunaite(
                model_id=model_sel,
                dataset_path=path,
                epochs=epochs,
                batch_size=1,
                grad_accum=4,
                quantization="4bit"
            )

            self.progress_bar.set(1.0)
            self.status_label.configure(text="✓ Training & Standalone Merge Complete!", text_color="#10b981")
            self.log(f"[SUCCESS] Standalone model saved. Final loss: {meta.get('final_loss', 0.0):.4f}")
            messagebox.showinfo("Complete", "Training and model merge finished successfully!")

        except Exception as e:
            self.status_label.configure(text="Training Failed", text_color="#f43f5e")
            self.log(f"[ERROR] {str(e)}")
            messagebox.showerror("Error", str(e))
        finally:
            self.is_training = False
            self.train_btn.configure(state="normal", text="⚡ TRAIN MODEL")


if __name__ == "__main__":
    app = LunaiteDesktopApp()
    app.mainloop()
