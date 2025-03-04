import tkinter as tk
from tkinter import filedialog, messagebox
import threading
import os
from main import process_image, process_video  # Import từ file chính

def select_input():
    filepath = filedialog.askopenfilename()
    if filepath:
        input_entry.delete(0, tk.END)
        input_entry.insert(0, filepath)

def select_output():
    filepath = filedialog.asksaveasfilename(defaultextension=".mp4" if mode_var.get() == "video" else ".jpg")
    if filepath:
        output_entry.delete(0, tk.END)
        output_entry.insert(0, filepath)

def start_processing():
    mode = mode_var.get()
    input_path = input_entry.get()
    output_path = output_entry.get()
    
    if not os.path.exists(input_path):
        messagebox.showerror("Error", "Input file does not exist!")
        return
    if not output_path:
        messagebox.showerror("Error", "Please select an output path!")
        return
    
    process_thread = threading.Thread(target=process_task, args=(mode, input_path, output_path))
    process_thread.start()

def process_task(mode, input_path, output_path):
    try:
        if mode == "image":
            process_image(input_path, output_path)
        else:
            process_video(input_path, output_path)
        messagebox.showinfo("Success", "Processing completed successfully!")
    except Exception as e:
        messagebox.showerror("Error", f"Processing failed: {str(e)}")

# GUI Setup
root = tk.Tk()
root.title("Object Removal using SwinSegFormer")
root.geometry("500x250")

mode_var = tk.StringVar(value="image")

tk.Label(root, text="Select Mode:").pack()
tk.Radiobutton(root, text="Image", variable=mode_var, value="image").pack()
tk.Radiobutton(root, text="Video", variable=mode_var, value="video").pack()

tk.Label(root, text="Input File:").pack()
input_entry = tk.Entry(root, width=50)
input_entry.pack()
tk.Button(root, text="Browse", command=select_input).pack()

tk.Label(root, text="Output File:").pack()
output_entry = tk.Entry(root, width=50)
output_entry.pack()
tk.Button(root, text="Save As", command=select_output).pack()

tk.Button(root, text="Start Processing", command=start_processing).pack()

root.mainloop()