import tkinter as tk
from tkinter import filedialog, messagebox
from PIL import Image, ImageTk
import threading
import os

# Function to select input file
def select_input():
    filepath = filedialog.askopenfilename(filetypes=[("Image Files", "*.jpg;*.png;*.jpeg"), ("All Files", "*.*")])
    if filepath:
        input_entry.delete(0, tk.END)
        input_entry.insert(0, filepath)
        display_image(filepath)

# Function to select output file
def select_output():
    filepath = filedialog.asksaveasfilename(defaultextension=".jpg", filetypes=[("JPEG", "*.jpg"), ("PNG", "*.png")])
    if filepath:
        output_entry.delete(0, tk.END)
        output_entry.insert(0, filepath)

# Function to display image preview
def display_image(filepath):
    img = Image.open(filepath)
    img = img.resize((400, 300), Image.Resampling.LANCZOS)
    img_tk = ImageTk.PhotoImage(img)
    image_label.config(image=img_tk)
    image_label.image = img_tk

# Function to start processing
def start_processing():
    input_path = input_entry.get()
    output_path = output_entry.get()
    
    if not os.path.exists(input_path):
        messagebox.showerror("Error", "Input file does not exist!")
        return
    if not output_path:
        messagebox.showerror("Error", "Please select an output path!")
        return
    
    process_thread = threading.Thread(target=process_task, args=(input_path, output_path))
    process_thread.start()

# Dummy function for processing
def process_task(input_path, output_path):
    messagebox.showinfo("Success", "Processing completed successfully!")

# GUI Setup
root = tk.Tk()
root.title("Object Removal using SwinSegFormer")
root.geometry("800x600")
root.configure(bg="white")

# Layout
frame_top = tk.Frame(root, bg="white")
frame_top.pack(pady=10)

frame_mid = tk.Frame(root, bg="white")
frame_mid.pack()

frame_bottom = tk.Frame(root, bg="white")
frame_bottom.pack(pady=20)

# Input selection
tk.Label(frame_top, text="Input File:", bg="white").grid(row=0, column=0, padx=5, pady=5)
input_entry = tk.Entry(frame_top, width=50)
input_entry.grid(row=0, column=1, padx=5, pady=5)
tk.Button(frame_top, text="Browse", command=select_input, bg="#3498db", fg="white").grid(row=0, column=2, padx=5, pady=5)

# Output selection
tk.Label(frame_top, text="Output File:", bg="white").grid(row=1, column=0, padx=5, pady=5)
output_entry = tk.Entry(frame_top, width=50)
output_entry.grid(row=1, column=1, padx=5, pady=5)
tk.Button(frame_top, text="Save As", command=select_output, bg="#2ecc71", fg="white").grid(row=1, column=2, padx=5, pady=5)

# Image preview
image_label = tk.Label(frame_mid, bg="white", width=400, height=300, relief="solid")
image_label.pack()

# Start processing button
tk.Button(frame_bottom, text="Start Processing", command=start_processing, font=("Arial", 12, "bold"), bg="#e67e22", fg="white", padx=20, pady=10).pack()

root.mainloop()




# def process_task(mode, input_path, output_path):
#     # try:
#     #     if mode == "image":
#     #         process_image(input_path, output_path)
#     #     else:
#     #         process_video(input_path, output_path)
#     #     messagebox.showinfo("Success", "Processing completed successfully!")
#     # except Exception as e:
#     #     messagebox.showerror("Error", f"Processing failed: {str(e)}")


