# complete_train_monitor.py
import tkinter as tk
import threading
import subprocess
import re
import time
import queue
import os
import sys
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import numpy as np
from collections import deque

class CompleteTrainingMonitor:
    def __init__(self):
        self.root = tk.Tk()
        self.root.title("Training Monitor with Charts")
        self.root.geometry("1200x800")
        
        # 使用Frame来确保明确的布局分区
        # 1. 顶部控制区 - 固定高度，确保按钮永远可见
        self.control_frame = tk.Frame(self.root, height=60, bg='navy')
        self.control_frame.pack(side=tk.TOP, fill=tk.X)
        self.control_frame.pack_propagate(False)  # 防止frame被内部widgets压缩
        
        # 突出显示的开始按钮 - 使用place确保精确位置
        self.start_button = tk.Button(
            self.control_frame, 
            text="START TRAINING", 
            command=self.start_training,
            width=20, height=1,
            bg="lime green", fg="white",
            font=("Arial", 12, "bold")
        )
        self.start_button.place(relx=0.25, rely=0.5, anchor=tk.CENTER)
        
        # 停止按钮
        self.stop_button = tk.Button(
            self.control_frame,
            text="STOP TRAINING",
            command=self.stop_training,
            width=20, height=1,
            bg="red", fg="white",
            state=tk.DISABLED,
            font=("Arial", 12, "bold")
        )
        self.stop_button.place(relx=0.5, rely=0.5, anchor=tk.CENTER)
        
        # 保存按钮
        self.save_button = tk.Button(
            self.control_frame,
            text="SAVE CHARTS",
            command=self.save_charts,
            width=20, height=1,
            bg="blue", fg="white",
            font=("Arial", 12, "bold")
        )
        self.save_button.place(relx=0.75, rely=0.5, anchor=tk.CENTER)
        
        # 2. 中部内容区 - 包含图表和日志
        self.content_frame = tk.Frame(self.root)
        self.content_frame.pack(side=tk.TOP, fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        # 左侧日志区域 - 占30%宽度
        self.log_frame = tk.LabelFrame(self.content_frame, text="Training Log", width=300)
        self.log_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=False, padx=5, pady=5)
        
        self.log_text = tk.Text(self.log_frame, wrap=tk.WORD, bg="black", fg="white", font=("Courier", 10))
        self.log_scroll = tk.Scrollbar(self.log_frame, command=self.log_text.yview)
        self.log_text.configure(yscrollcommand=self.log_scroll.set)
        
        self.log_text.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        self.log_scroll.pack(side=tk.RIGHT, fill=tk.Y)
        
        # 右侧图表区域 - 占70%宽度
        self.chart_frame = tk.LabelFrame(self.content_frame, text="Training Metrics")
        self.chart_frame.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        # 创建图表
        self.fig = plt.figure(figsize=(8, 8))
        
        # 奖励曲线
        self.ax1 = self.fig.add_subplot(321)
        self.reward_line, = self.ax1.plot([], [], 'b-', marker='o', markersize=2, label='Reward')
        self.ax1.set_title('Average Reward')
        self.ax1.set_xlabel('Episode')
        self.ax1.set_ylabel('Reward')
        self.ax1.grid(True)
        
        # 风险曲线
        self.ax2 = self.fig.add_subplot(322)
        self.risk_line, = self.ax2.plot([], [], 'r-', marker='o', markersize=2, label='Risk')
        self.ax2.set_title('Average Risk')
        self.ax2.set_xlabel('Episode')
        self.ax2.set_ylabel('Risk Index')
        self.ax2.grid(True)
        
        # 碰撞曲线
        self.ax3 = self.fig.add_subplot(323)
        self.collision_line, = self.ax3.plot([], [], 'g-', marker='o', markersize=2, label='Collisions')
        self.ax3.set_title('Collision Count')
        self.ax3.set_xlabel('Episode')
        self.ax3.set_ylabel('Collision Count')
        self.ax3.grid(True)
        
        # 变道曲线
        self.ax4 = self.fig.add_subplot(324)
        self.lane_change_line, = self.ax4.plot([], [], 'y-', marker='o', markersize=2, label='Lane Changes')
        self.ax4.set_title('Lane Change Count')
        self.ax4.set_xlabel('Episode')
        self.ax4.set_ylabel('Lane Change Count')
        self.ax4.grid(True)
        
        # 速度曲线
        self.ax5 = self.fig.add_subplot(325)
        self.speed_line, = self.ax5.plot([], [], 'c-', marker='o', markersize=2, label='Speed')
        self.ax5.set_title('Average Speed')
        self.ax5.set_xlabel('Episode')
        self.ax5.set_ylabel('Speed (m/s)')
        self.ax5.grid(True)
        
        # TTC曲线
        self.ax6 = self.fig.add_subplot(326)
        self.ttc_line, = self.ax6.plot([], [], 'm-', marker='o', markersize=2, label='TTC')
        self.ax6.set_title('Time To Collision')
        self.ax6.set_xlabel('Episode')
        self.ax6.set_ylabel('TTC (s)')
        self.ax6.grid(True)
        
        self.fig.tight_layout()
        
        # 添加图表到Tkinter
        self.canvas = FigureCanvasTkAgg(self.fig, master=self.chart_frame)
        self.canvas.draw()
        self.canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
        
        # 3. 底部状态栏
        self.status_var = tk.StringVar()
        self.status_var.set("Ready - Click the GREEN button to start training")
        self.status_label = tk.Label(
            self.root,
            textvariable=self.status_var,
            bd=1, relief=tk.SUNKEN,
            font=("Arial", 10, "bold"),
            bg="lightgray", fg="black"
        )
        self.status_label.pack(side=tk.BOTTOM, fill=tk.X)
        
        # 训练数据
        self.episode_counter = 0
        self.episodes = []
        self.rewards = []
        self.risks = []
        self.collisions = []
        self.lane_changes = []
        self.speeds = []
        self.ttcs = []
        
        # 用于滚动平均的队列
        self.window_size = 10  # 滚动平均窗口大小
        self.reward_queue = deque(maxlen=self.window_size)
        self.risk_queue = deque(maxlen=self.window_size)
        self.collision_queue = deque(maxlen=self.window_size)
        self.lane_change_queue = deque(maxlen=self.window_size)
        self.speed_queue = deque(maxlen=self.window_size)
        self.ttc_queue = deque(maxlen=self.window_size)
        
        # 线程和进程控制
        self.training_process = None
        self.running = False
        self.message_queue = queue.Queue()
        
        # 记录间隔
        self.record_interval = 100
        
        # 创建结果目录
        os.makedirs("results", exist_ok=True)
        
        # 添加初始消息
        self.log_text.insert(tk.END, "=== TRAINING MONITOR INITIALIZED ===\n\n")
        self.log_text.insert(tk.END, "Click the GREEN 'START TRAINING' button to begin.\n")
        self.log_text.insert(tk.END, "Charts will update every 100 episodes.\n")
        self.log_text.see(tk.END)
        
        # 开始UI更新循环
        self.update_ui()
    
    def start_training(self):
        """开始训练过程"""
        if self.running:
            return
        
        self.running = True
        self.start_button.config(state=tk.DISABLED)
        self.stop_button.config(state=tk.NORMAL)
        self.status_var.set("Training in progress...")
        
        self.log_text.insert(tk.END, "\n=== TRAINING STARTED ===\n\n")
        self.log_text.see(tk.END)
        
        # 创建临时脚本处理编码问题
        with open("run_train_utf8.py", "w", encoding="utf-8") as f:
            f.write("""
# Temporary script to run main_train.py with proper encoding
import os
import sys
import subprocess

# Set environment variable for encoding
os.environ["PYTHONIOENCODING"] = "utf-8"

# Modify environment configuration to disable visual rendering
os.environ["HIGHWAY_ENV_DISABLE_RENDER"] = "1"

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
        
        # 在单独的线程中启动训练
        def run_training():
            try:
                # 使用编码安全的脚本
                self.training_process = subprocess.Popen(
                    ["python", "run_train_utf8.py"],
                    stdout=subprocess.PIPE,
                    stderr=subprocess.STDOUT,
                    universal_newlines=True,
                    encoding="utf-8",  # 显式使用UTF-8
                    errors="replace",  # 替换无效字符
                    bufsize=1
                )
                
                # 读取并过滤输出
                for line in iter(self.training_process.stdout.readline, ''):
                    # 过滤无关行
                    if any(excluded in line for excluded in ["crash", "over"]) or not line.strip():
                        continue
                    
                    # 添加到消息队列
                    self.message_queue.put(line)
                    
                    # 解析训练数据
                    self.parse_metrics(line)
                    
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
                
                # 清理临时脚本
                try:
                    os.remove("run_train_utf8.py")
                except:
                    pass
        
        # 启动线程
        threading.Thread(target=run_training, daemon=True).start()
    
    def stop_training(self):
        """停止训练过程"""
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
    
    def save_charts(self):
        """保存图表到文件"""
        try:
            # 如果没有数据，则不保存
            if not self.episodes:
                self.log_text.insert(tk.END, "No data to save.\n")
                return
                
            # 保存各个图表
            timestamp = time.strftime("%Y%m%d_%H%M%S")
            self.fig.savefig(f"results/training_charts_{timestamp}.png", dpi=150)
            
            # 保存数据为CSV
            with open(f"results/training_data_{timestamp}.csv", 'w', encoding='utf-8') as f:
                f.write("Episode,Reward,Risk,Collisions,LaneChanges,Speed,TTC\n")
                for i in range(len(self.episodes)):
                    row = [
                        self.episodes[i],
                        self.rewards[i] if i < len(self.rewards) else "",
                        self.risks[i] if i < len(self.risks) else "",
                        self.collisions[i] if i < len(self.collisions) else "",
                        self.lane_changes[i] if i < len(self.lane_changes) else "",
                        self.speeds[i] if i < len(self.speeds) else "",
                        self.ttcs[i] if i < len(self.ttcs) else ""
                    ]
                    f.write(",".join(map(str, row)) + "\n")
                
            self.log_text.insert(tk.END, f"\nCharts and data saved to results/ folder with timestamp {timestamp}\n")
            self.log_text.see(tk.END)
        except Exception as e:
            self.log_text.insert(tk.END, f"\nError saving charts: {str(e)}\n")
            self.log_text.see(tk.END)
    
    def parse_metrics(self, line):
        """解析训练指标"""
        try:
            # 匹配Episode行
            episode_match = re.search(r'Episode (\d+)/(\d+)', line)
            reward_match = re.search(r'Reward: ([-+]?\d*\.\d+|\d+)', line)
            risk_match = re.search(r'Risk: ([-+]?\d*\.\d+|\d+)', line)
            lane_match = re.search(r'LaneChanges: (\d+)', line)
            collision_match = re.search(r'Collisions: (\d+)', line)
            
            # 尝试匹配速度和TTC（假设这些在日志中）
            speed_match = re.search(r'Speed: ([-+]?\d*\.\d+|\d+)', line)
            ttc_match = re.search(r'TTC: ([-+]?\d*\.\d+|\d+)', line)
            
            if episode_match and reward_match:
                episode = int(episode_match.group(1))
                
                # 如果新的episode不等于预期的下一个episode，可能存在重置
                if self.episodes and episode <= self.episodes[-1]:
                    self.log_text.insert(tk.END, "\n=== TRAINING RESET DETECTED - CLEARING DATA ===\n")
                    self.episodes = []
                    self.rewards = []
                    self.risks = []
                    self.collisions = []
                    self.lane_changes = []
                    self.speeds = []
                    self.ttcs = []
                    self.reward_queue.clear()
                    self.risk_queue.clear()
                    self.collision_queue.clear()
                    self.lane_change_queue.clear()
                    self.speed_queue.clear()
                    self.ttc_queue.clear()
                
                reward = float(reward_match.group(1))
                risk = float(risk_match.group(1)) if risk_match else 0
                collisions = int(collision_match.group(1)) if collision_match else 0
                lane_changes = int(lane_match.group(1)) if lane_match else 0
                speed = float(speed_match.group(1)) if speed_match else 0
                ttc = float(ttc_match.group(1)) if ttc_match else 0
                
                # 添加值到滚动队列
                self.reward_queue.append(reward)
                self.risk_queue.append(risk)
                self.collision_queue.append(collisions)
                self.lane_change_queue.append(lane_changes)
                self.speed_queue.append(speed)
                self.ttc_queue.append(ttc)
                
                # 每100个episode记录一次（使用滚动窗口的平均值）
                if episode % self.record_interval == 0:
                    # 计算滚动平均
                    avg_reward = sum(self.reward_queue) / len(self.reward_queue) if self.reward_queue else reward
                    avg_risk = sum(self.risk_queue) / len(self.risk_queue) if self.risk_queue else risk
                    avg_collisions = sum(self.collision_queue) / len(self.collision_queue) if self.collision_queue else collisions
                    avg_lane_changes = sum(self.lane_change_queue) / len(self.lane_change_queue) if self.lane_change_queue else lane_changes
                    avg_speed = sum(self.speed_queue) / len(self.speed_queue) if self.speed_queue else speed
                    avg_ttc = sum(self.ttc_queue) / len(self.ttc_queue) if self.ttc_queue else ttc
                    
                    self.episodes.append(episode)
                    self.rewards.append(avg_reward)
                    self.risks.append(avg_risk)
                    self.collisions.append(avg_collisions)
                    self.lane_changes.append(avg_lane_changes)
                    self.speeds.append(avg_speed)
                    self.ttcs.append(avg_ttc)
                    
                    # 记录到日志
                    self.log_text.insert(tk.END, f"\n=== RECORDING DATA AT EPISODE {episode} ===\n")
                    self.log_text.insert(tk.END, f"Reward: {avg_reward:.2f}, Risk: {avg_risk:.2f}, Collisions: {avg_collisions:.2f}\n")
                    self.log_text.insert(tk.END, f"Lane Changes: {avg_lane_changes:.2f}, Speed: {avg_speed:.2f}, TTC: {avg_ttc:.2f}\n")
                    self.log_text.see(tk.END)
        except Exception as e:
            print(f"Parsing error: {e}")
    
    def update_ui(self):
        """更新UI界面"""
        # 处理消息队列
        try:
            while True:
                message = self.message_queue.get_nowait()
                
                if message == "__TRAINING_FINISHED__":
                    self.start_button.config(state=tk.NORMAL)
                    self.stop_button.config(state=tk.DISABLED)
                    self.status_var.set("Training completed")
                    self.log_text.insert(tk.END, "\n=== TRAINING FINISHED ===\n")
                    self.log_text.see(tk.END)
                    
                    # 保存最终图表
                    self.save_charts()
                    continue
                
                # 添加到日志
                self.log_text.insert(tk.END, message)
                self.log_text.see(tk.END)
                
                # 从episode信息更新状态
                if "Episode" in message and "Reward" in message:
                    self.status_var.set(message.strip())
        except queue.Empty:
            pass
        
        # 更新图表
        if self.episodes:
            # 更新奖励曲线
            self.reward_line.set_data(self.episodes, self.rewards)
            self.ax1.relim()
            self.ax1.autoscale_view()
            
            # 更新风险曲线
            self.risk_line.set_data(self.episodes, self.risks)
            self.ax2.relim()
            self.ax2.autoscale_view()
            
            # 更新碰撞曲线
            self.collision_line.set_data(self.episodes, self.collisions)
            self.ax3.relim()
            self.ax3.autoscale_view()
            
            # 更新变道曲线
            self.lane_change_line.set_data(self.episodes, self.lane_changes)
            self.ax4.relim()
            self.ax4.autoscale_view()
            
            # 更新速度曲线
            self.speed_line.set_data(self.episodes, self.speeds)
            self.ax5.relim()
            self.ax5.autoscale_view()
            
            # 更新TTC曲线
            self.ttc_line.set_data(self.episodes, self.ttcs)
            self.ax6.relim()
            self.ax6.autoscale_view()
            
            # 重绘图表
            self.canvas.draw()
        
        # 每100ms更新一次
        self.root.after(100, self.update_ui)
    
    def run(self):
        """运行主循环"""
        self.root.mainloop()

if __name__ == "__main__":
    monitor = CompleteTrainingMonitor()
    monitor.run()