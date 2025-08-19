# simple_train_monitor_fixed.py
import tkinter as tk
import threading
import subprocess
import re
import time
import queue
import os
import sys

class SimpleTrainingMonitor:
    def __init__(self):
        self.root = tk.Tk()
        self.root.title("Simple RL Training Monitor")
        self.root.geometry("800x600")
        
        # Create simplified layout
        # Top controls
        self.control_frame = tk.Frame(self.root, bg="lightgray", height=50)
        self.control_frame.pack(fill=tk.X, padx=10, pady=10)
        
        # Start button
        self.start_button = tk.Button(
            self.control_frame, 
            text="START TRAINING", 
            command=self.start_training,
            width=20, height=2,
            bg="green", fg="white",
            font=("Arial", 10, "bold")
        )
        self.start_button.pack(side=tk.LEFT, padx=10, pady=5)
        
        # Stop button
        self.stop_button = tk.Button(
            self.control_frame,
            text="STOP TRAINING",
            command=self.stop_training,
            width=20, height=2,
            bg="red", fg="white",
            state=tk.DISABLED,
            font=("Arial", 10, "bold")
        )
        self.stop_button.pack(side=tk.LEFT, padx=10, pady=5)
        
        # Clear button
        self.clear_button = tk.Button(
            self.control_frame,
            text="CLEAR LOG",
            command=self.clear_log,
            width=15, height=2,
            font=("Arial", 10)
        )
        self.clear_button.pack(side=tk.LEFT, padx=10, pady=5)
        
        # Status text
        self.status_var = tk.StringVar()
        self.status_var.set("Ready to start training")
        self.status_label = tk.Label(
            self.root,
            textvariable=self.status_var,
            bd=1, relief=tk.SUNKEN,
            font=("Arial", 10),
            anchor=tk.W
        )
        self.status_label.pack(fill=tk.X, side=tk.BOTTOM, padx=10, pady=5)
        
        # Log area
        self.log_frame = tk.Frame(self.root)
        self.log_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=5)
        
        self.log_text = tk.Text(self.log_frame, wrap=tk.WORD, bg="black", fg="white")
        self.log_scroll = tk.Scrollbar(self.log_frame, command=self.log_text.yview)
        self.log_text.configure(yscrollcommand=self.log_scroll.set)
        
        self.log_text.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        self.log_scroll.pack(side=tk.RIGHT, fill=tk.Y)
        
        # Add initial messages
        self.log_text.insert(tk.END, "=== TRAINING MONITOR INITIALIZED ===\n\n")
        self.log_text.insert(tk.END, "Click the GREEN 'START TRAINING' button at the top to begin.\n")
        self.log_text.see(tk.END)
        
        # Training variables
        self.training_process = None
        self.running = False
        self.message_queue = queue.Queue()
        
        # Start UI update loop
        self.update_ui()
    
    def start_training(self):
        """Start the training process"""
        if self.running:
            return
        
        self.running = True
        self.start_button.config(state=tk.DISABLED)
        self.stop_button.config(state=tk.NORMAL)
        self.status_var.set("Training in progress...")
        
        self.log_text.insert(tk.END, "\n=== TRAINING STARTED ===\n\n")
        self.log_text.see(tk.END)
        
        # Create temporary script with explicit encoding
        with open("run_train_utf8.py", "w", encoding="utf-8") as f:
            f.write("""
# Temporary script to run main_train.py with proper encoding
import os
import sys
import subprocess

# Set environment variable for encoding
os.environ["PYTHONIOENCODING"] = "utf-8"

# Run the main training script
try:
    # Force UTF-8 encoding for all I/O
    process = subprocess.Popen(
        ["python", "main_train.py"],
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        universal_newlines=True,
        encoding="utf-8",
        errors="replace",  # Replace invalid chars instead of crashing
        bufsize=1
    )
    
    for line in process.stdout:
        sys.stdout.write(line)
        sys.stdout.flush()
        
    process.wait()
    sys.exit(process.returncode)
except Exception as e:
    print(f"Error executing main_train.py: {e}")
    sys.exit(1)
""")
        
        # Start training in a separate thread
        def run_training():
            try:
                # Use our encoding-safe script
                self.training_process = subprocess.Popen(
                    ["python", "run_train_utf8.py"],
                    stdout=subprocess.PIPE,
                    stderr=subprocess.STDOUT,
                    universal_newlines=True,
                    encoding="utf-8",  # Explicitly use UTF-8
                    errors="replace",  # Replace invalid characters
                    bufsize=1
                )
                
                # Read and filter output
                for line in iter(self.training_process.stdout.readline, ''):
                    # Filter out unwanted lines
                    if any(excluded in line for excluded in ["crash", "over"]) or not line.strip():
                        continue
                    
                    # Add to message queue
                    self.message_queue.put(line)
                    
                    if not self.running:
                        self.training_process.terminate()
                        break
                
                self.training_process.wait()
                self.message_queue.put("Training completed")
                
            except Exception as e:
                self.message_queue.put(f"Error: {str(e)}")
                import traceback
                self.message_queue.put(traceback.format_exc())
            finally:
                self.running = False
                self.message_queue.put("__TRAINING_FINISHED__")
                
                # Clean up temporary script
                try:
                    os.remove("run_train_utf8.py")
                except:
                    pass
        
        # Start thread
        threading.Thread(target=run_training, daemon=True).start()
    
    def stop_training(self):
        """Stop the training process"""
        if not self.running:
            return
        
        self.running = False
        self.status_var.set("Stopping training...")
        self.log_text.insert(tk.END, "\n=== STOPPING TRAINING ===\n")
        self.log_text.see(tk.END)
        
        if self.training_process:
            try:
                self.training_process.terminate()
            except:
                pass
    
    def clear_log(self):
        """Clear the log text area"""
        self.log_text.delete(1.0, tk.END)
        self.log_text.insert(tk.END, "Log cleared.\n")
    
    def update_ui(self):
        """Update the UI with new messages"""
        try:
            while True:
                message = self.message_queue.get_nowait()
                
                if message == "__TRAINING_FINISHED__":
                    self.start_button.config(state=tk.NORMAL)
                    self.stop_button.config(state=tk.DISABLED)
                    self.status_var.set("Training completed")
                    self.log_text.insert(tk.END, "\n=== TRAINING FINISHED ===\n")
                    self.log_text.see(tk.END)
                    continue
                
                # Add message to log
                self.log_text.insert(tk.END, message)
                self.log_text.see(tk.END)
                
                # Update status from episode info
                if "Episode" in message and "Reward" in message:
                    self.status_var.set(message.strip())
        except queue.Empty:
            pass
        
        # Schedule next update
        self.root.after(100, self.update_ui)
    
    def run(self):
        """Run the main UI loop"""
        self.root.mainloop()

if __name__ == "__main__":
    monitor = SimpleTrainingMonitor()
    monitor.run()