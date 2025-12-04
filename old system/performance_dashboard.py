"""
Comprehensive Agent Performance Dashboard
Advanced monitoring with trends, alerts, and detailed analytics
"""

import psutil
import time
import os
import json
from datetime import datetime, timedelta
from collections import deque
from typing import Dict, List, Optional
import statistics

class AgentPerformanceDashboard:
    """Advanced performance monitoring dashboard"""
    
    def __init__(self, history_size: int = 30):
        self.history_size = history_size
        self.performance_history = {
            "Policy Analysis": deque(maxlen=history_size),
            "Application Processing": deque(maxlen=history_size), 
            "Decision Making": deque(maxlen=history_size)
        }
        self.alerts = []
        self.start_time = datetime.now()
        
        # Performance thresholds
        self.cpu_alert_threshold = 80.0  # %
        self.memory_alert_threshold = 500.0  # MB
        self.memory_critical_threshold = 1000.0  # MB
        
    def find_agent_processes(self) -> Dict[str, Optional[psutil.Process]]:
        """Find agent processes"""
        agents = {
            "Policy Analysis": None,
            "Application Processing": None,
            "Decision Making": None
        }
        
        for proc in psutil.process_iter(['pid', 'name', 'cmdline', 'create_time']):
            try:
                if proc.info['name'] in ['python.exe', 'python']:
                    cmdline = ' '.join(proc.info['cmdline'] or [])
                    
                    if '--port 10001' in cmdline or 'policy-analysis-agent' in cmdline:
                        agents["Policy Analysis"] = proc
                    elif '--port 10002' in cmdline or 'application-processing-agent' in cmdline:
                        agents["Application Processing"] = proc
                    elif '--port 10003' in cmdline or 'decision-making-agent' in cmdline:
                        agents["Decision Making"] = proc
                        
            except (psutil.NoSuchProcess, psutil.AccessDenied):
                continue
        
        return agents
    
    def get_process_stats(self, proc: psutil.Process) -> Dict:
        """Get detailed process statistics"""
        try:
            memory_info = proc.memory_info()
            cpu_percent = proc.cpu_percent()
            
            # Get additional details
            num_threads = proc.num_threads()
            open_files = len(proc.open_files())
            connections = len(proc.connections())
            
            return {
                "timestamp": datetime.now(),
                "pid": proc.pid,
                "cpu_percent": cpu_percent,
                "memory_mb": memory_info.rss / 1024 / 1024,
                "memory_vms_mb": memory_info.vms / 1024 / 1024,
                "num_threads": num_threads,
                "open_files": open_files,
                "connections": connections,
                "status": "RUNNING"
            }
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            return {
                "timestamp": datetime.now(),
                "pid": None,
                "cpu_percent": 0,
                "memory_mb": 0,
                "memory_vms_mb": 0,
                "num_threads": 0,
                "open_files": 0,
                "connections": 0,
                "status": "STOPPED"
            }
    
    def check_alerts(self, agent_name: str, stats: Dict) -> List[str]:
        """Check for performance alerts"""
        alerts = []
        
        if stats["status"] == "STOPPED":
            alerts.append(f"🔴 {agent_name} is STOPPED")
        elif stats["status"] == "RUNNING":
            if stats["cpu_percent"] > self.cpu_alert_threshold:
                alerts.append(f"⚠️ {agent_name} high CPU: {stats['cpu_percent']:.1f}%")
            
            if stats["memory_mb"] > self.memory_critical_threshold:
                alerts.append(f"🚨 {agent_name} CRITICAL memory: {stats['memory_mb']:.1f}MB")
            elif stats["memory_mb"] > self.memory_alert_threshold:
                alerts.append(f"⚠️ {agent_name} high memory: {stats['memory_mb']:.1f}MB")
            
            if stats["num_threads"] > 50:
                alerts.append(f"⚠️ {agent_name} high thread count: {stats['num_threads']}")
        
        return alerts
    
    def calculate_trends(self, agent_name: str) -> Dict:
        """Calculate performance trends"""
        history = list(self.performance_history[agent_name])
        if len(history) < 2:
            return {"cpu_trend": "stable", "memory_trend": "stable"}
        
        # Get recent data points
        recent_cpu = [h["cpu_percent"] for h in history[-10:]]
        recent_memory = [h["memory_mb"] for h in history[-10:]]
        
        # Calculate trends
        cpu_trend = "stable"
        memory_trend = "stable"
        
        if len(recent_cpu) >= 5:
            cpu_slope = (recent_cpu[-1] - recent_cpu[0]) / len(recent_cpu)
            if cpu_slope > 2:
                cpu_trend = "increasing"
            elif cpu_slope < -2:
                cpu_trend = "decreasing"
        
        if len(recent_memory) >= 5:
            memory_slope = (recent_memory[-1] - recent_memory[0]) / len(recent_memory)
            if memory_slope > 10:  # MB
                memory_trend = "increasing"
            elif memory_slope < -10:
                memory_trend = "decreasing"
        
        return {
            "cpu_trend": cpu_trend,
            "memory_trend": memory_trend,
            "avg_cpu": statistics.mean(recent_cpu) if recent_cpu else 0,
            "avg_memory": statistics.mean(recent_memory) if recent_memory else 0
        }
    
    def format_bytes(self, bytes_val: float) -> str:
        """Format bytes in human readable format"""
        for unit in ['B', 'KB', 'MB', 'GB']:
            if bytes_val < 1024.0:
                return f"{bytes_val:.1f} {unit}"
            bytes_val /= 1024.0
        return f"{bytes_val:.1f} TB"
    
    def format_uptime(self, seconds: float) -> str:
        """Format uptime"""
        hours = int(seconds // 3600)
        minutes = int((seconds % 3600) // 60)
        secs = int(seconds % 60)
        return f"{hours:02d}:{minutes:02d}:{secs:02d}"
    
    def clear_screen(self):
        """Clear screen"""
        os.system('cls' if os.name == 'nt' else 'clear')
    
    def print_dashboard(self):
        """Print the main dashboard"""
        self.clear_screen()
        
        # Header
        print("="*100)
        print("🎯 PRIOR AUTHORIZATION AGENTS - PERFORMANCE DASHBOARD")
        print("="*100)
        
        current_time = datetime.now()
        uptime = current_time - self.start_time
        
        print(f"Dashboard Started: {self.start_time.strftime('%Y-%m-%d %H:%M:%S')} | "
              f"Uptime: {self.format_uptime(uptime.total_seconds())} | "
              f"Last Updated: {current_time.strftime('%H:%M:%S')}")
        
        # System overview
        cpu_count = psutil.cpu_count()
        system_memory = psutil.virtual_memory()
        disk_usage = psutil.disk_usage('/')
        
        print(f"System: {cpu_count} CPUs @ {psutil.cpu_percent():.1f}% | "
              f"RAM: {self.format_bytes(system_memory.used)}/{self.format_bytes(system_memory.total)} "
              f"({system_memory.percent:.1f}%) | "
              f"Disk: {disk_usage.percent:.1f}%")
        
        print("-"*100)
        
        # Agent statistics
        agents = self.find_agent_processes()
        
        print(f"{'AGENT':<20} {'STATUS':<10} {'PID':<8} {'CPU%':<8} {'MEMORY':<12} {'THREADS':<8} {'FILES':<8} {'TREND':<15}")
        print("-"*100)
        
        all_alerts = []
        
        for agent_name, proc in agents.items():
            if proc is None:
                stats = {
                    "timestamp": datetime.now(),
                    "pid": None,
                    "cpu_percent": 0,
                    "memory_mb": 0,
                    "memory_vms_mb": 0,
                    "num_threads": 0,
                    "open_files": 0,
                    "connections": 0,
                    "status": "STOPPED"
                }
            else:
                stats = self.get_process_stats(proc)
            
            # Add to history
            self.performance_history[agent_name].append(stats)
            
            # Check alerts
            agent_alerts = self.check_alerts(agent_name, stats)
            all_alerts.extend(agent_alerts)
            
            # Calculate trends
            trends = self.calculate_trends(agent_name)
            
            # Format display
            status_icon = "🟢" if stats["status"] == "RUNNING" else "🔴"
            pid_str = str(stats["pid"]) if stats["pid"] else "N/A"
            memory_str = f"{stats['memory_mb']:.1f}MB"
            trend_str = f"CPU:{trends['cpu_trend'][:3]} MEM:{trends['memory_trend'][:3]}"
            
            print(f"{agent_name:<18} {status_icon} {stats['status']:<8} {pid_str:<8} "
                  f"{stats['cpu_percent']:<8.1f} {memory_str:<12} {stats['num_threads']:<8} "
                  f"{stats['open_files']:<8} {trend_str:<15}")
        
        # Summary
        running_agents = sum(1 for proc in agents.values() if proc is not None)
        total_cpu = sum(self.get_process_stats(proc)["cpu_percent"] for proc in agents.values() if proc is not None)
        total_memory = sum(self.get_process_stats(proc)["memory_mb"] for proc in agents.values() if proc is not None)
        
        print("-"*100)
        print(f"SUMMARY: {running_agents}/3 agents running | "
              f"Total CPU: {total_cpu:.1f}% | "
              f"Total Memory: {total_memory:.1f}MB | "
              f"Alerts: {len(all_alerts)}")
        
        # Alerts section
        if all_alerts:
            print("\n🚨 ACTIVE ALERTS:")
            for alert in all_alerts[-5:]:  # Show last 5 alerts
                print(f"  {alert}")
        
        print("\n" + "="*100)
        print("Press Ctrl+C to exit | Refresh: 3s | Use 'python monitor_agents.py --help' for options")
    
    def run(self, refresh_interval: float = 3.0):
        """Run the dashboard"""
        try:
            while True:
                self.print_dashboard()
                time.sleep(refresh_interval)
                
        except KeyboardInterrupt:
            print("\n\nDashboard stopped.")
        except Exception as e:
            print(f"\nError: {e}")

def main():
    """Main function"""
    print("🚀 Starting Performance Dashboard...")
    print("Waiting 5 seconds for agents to initialize...\n")
    time.sleep(5)
    
    dashboard = AgentPerformanceDashboard()
    dashboard.run()

if __name__ == "__main__":
    main()