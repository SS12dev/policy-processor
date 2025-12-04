"""
Simple standalone agent monitor
Quick monitoring script that can be run independently
"""

import psutil
import time
import os
from datetime import datetime

def find_agent_processes():
    """Find agent processes by looking for Python processes with specific ports"""
    agents = {
        "Policy Analysis (10001)": None,
        "Application Processing (10002)": None, 
        "Decision Making (10003)": None
    }
    
    for proc in psutil.process_iter(['pid', 'name', 'cmdline']):
        try:
            if proc.info['name'] in ['python.exe', 'python']:
                cmdline = ' '.join(proc.info['cmdline'] or [])
                
                if '--port 10001' in cmdline or 'policy-analysis-agent' in cmdline:
                    agents["Policy Analysis (10001)"] = proc
                elif '--port 10002' in cmdline or 'application-processing-agent' in cmdline:
                    agents["Application Processing (10002)"] = proc
                elif '--port 10003' in cmdline or 'decision-making-agent' in cmdline:
                    agents["Decision Making (10003)"] = proc
                    
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            continue
    
    return agents

def format_bytes(bytes_val):
    """Format bytes in human readable format"""
    for unit in ['B', 'KB', 'MB', 'GB']:
        if bytes_val < 1024.0:
            return f"{bytes_val:.1f} {unit}"
        bytes_val /= 1024.0
    return f"{bytes_val:.1f} TB"

def main():
    print("Simple Agent Monitor - Starting...")
    print("Press Ctrl+C to stop\n")
    
    try:
        while True:
            os.system('cls' if os.name == 'nt' else 'clear')
            
            print("="*70)
            print(f"AGENT MONITOR - {datetime.now().strftime('%H:%M:%S')}")
            print("="*70)
            
            agents = find_agent_processes()
            
            print(f"{'AGENT':<30} {'STATUS':<10} {'CPU%':<8} {'MEMORY'}")
            print("-"*70)
            
            total_cpu = 0
            total_memory = 0
            running_count = 0
            
            for agent_name, proc in agents.items():
                if proc is None:
                    print(f"{agent_name:<30} {'🔴 STOPPED':<10} {'0.0':<8} {'0 MB'}")
                else:
                    try:
                        cpu = proc.cpu_percent()
                        memory = proc.memory_info().rss
                        memory_str = format_bytes(memory)
                        
                        total_cpu += cpu
                        total_memory += memory
                        running_count += 1
                        
                        print(f"{agent_name:<30} {'🟢 RUNNING':<10} {cpu:<8.1f} {memory_str}")
                    except (psutil.NoSuchProcess, psutil.AccessDenied):
                        print(f"{agent_name:<30} {'🟡 ERROR':<10} {'N/A':<8} {'N/A'}")
            
            print("-"*70)
            print(f"SUMMARY: {running_count}/3 running | "
                  f"Total CPU: {total_cpu:.1f}% | "
                  f"Total Memory: {format_bytes(total_memory)}")
            print("="*70)
            
            time.sleep(2)
            
    except KeyboardInterrupt:
        print("\n\nMonitoring stopped.")

if __name__ == "__main__":
    main()