"""
Real-time monitoring system for Prior Authorization Agents
Monitors CPU and memory usage of all three agents with live updates
"""

import psutil
import time
import os
import sys
from datetime import datetime
from typing import Dict, List, Optional
import argparse
from dataclasses import dataclass
import json

@dataclass
class AgentStats:
    """Statistics for a single agent"""
    name: str
    pid: Optional[int]
    port: int
    cpu_percent: float
    memory_mb: float
    memory_percent: float
    status: str
    uptime_seconds: float
    start_time: Optional[datetime]

class AgentMonitor:
    """Monitor for Prior Authorization Agents"""
    
    def __init__(self, refresh_interval: float = 2.0):
        self.refresh_interval = refresh_interval
        self.agents_config = {
            10001: {"name": "Policy Analysis", "process_name": "python"},
            10002: {"name": "Application Processing", "process_name": "python"},
            10003: {"name": "Decision Making", "process_name": "python"}
        }
        self.start_time = datetime.now()
        self.stats_history = []
        
    def find_agent_processes(self) -> Dict[int, psutil.Process]:
        """Find running agent processes by port"""
        agent_processes = {}
        
        for proc in psutil.process_iter(['pid', 'name', 'cmdline', 'create_time']):
            try:
                # Check if it's a Python process with agent-related command line
                if proc.info['name'] == 'python.exe' or proc.info['name'] == 'python':
                    cmdline = ' '.join(proc.info['cmdline'] or [])
                    
                    # Check for each agent by looking for port numbers or paths
                    for port, config in self.agents_config.items():
                        if (f"--port {port}" in cmdline or 
                            f"port {port}" in cmdline or
                            f":{port}" in cmdline or
                            "policy-analysis-agent" in cmdline and port == 10001 or
                            "application-processing-agent" in cmdline and port == 10002 or
                            "decision-making-agent" in cmdline and port == 10003):
                            
                            agent_processes[port] = proc
                            break
                            
            except (psutil.NoSuchProcess, psutil.AccessDenied, psutil.ZombieProcess):
                continue
                
        return agent_processes
    
    def get_agent_stats(self, port: int, process: Optional[psutil.Process]) -> AgentStats:
        """Get statistics for a single agent"""
        config = self.agents_config[port]
        
        if process is None:
            return AgentStats(
                name=config["name"],
                pid=None,
                port=port,
                cpu_percent=0.0,
                memory_mb=0.0,
                memory_percent=0.0,
                status="STOPPED",
                uptime_seconds=0.0,
                start_time=None
            )
        
        try:
            # Get process statistics
            memory_info = process.memory_info()
            memory_mb = memory_info.rss / 1024 / 1024  # Convert to MB
            memory_percent = process.memory_percent()
            cpu_percent = process.cpu_percent()
            
            # Calculate uptime
            create_time = datetime.fromtimestamp(process.create_time())
            uptime_seconds = (datetime.now() - create_time).total_seconds()
            
            return AgentStats(
                name=config["name"],
                pid=process.pid,
                port=port,
                cpu_percent=cpu_percent,
                memory_mb=memory_mb,
                memory_percent=memory_percent,
                status="RUNNING",
                uptime_seconds=uptime_seconds,
                start_time=create_time
            )
            
        except (psutil.NoSuchProcess, psutil.AccessDenied, psutil.ZombieProcess):
            return AgentStats(
                name=config["name"],
                pid=None,
                port=port,
                cpu_percent=0.0,
                memory_mb=0.0,
                memory_percent=0.0,
                status="ERROR",
                uptime_seconds=0.0,
                start_time=None
            )
    
    def format_uptime(self, seconds: float) -> str:
        """Format uptime in human-readable format"""
        if seconds == 0:
            return "00:00:00"
        
        hours = int(seconds // 3600)
        minutes = int((seconds % 3600) // 60)
        secs = int(seconds % 60)
        return f"{hours:02d}:{minutes:02d}:{secs:02d}"
    
    def format_memory(self, mb: float) -> str:
        """Format memory usage in human-readable format"""
        if mb >= 1024:
            return f"{mb/1024:.1f} GB"
        else:
            return f"{mb:.1f} MB"
    
    def clear_screen(self):
        """Clear the terminal screen"""
        os.system('cls' if os.name == 'nt' else 'clear')
    
    def print_header(self):
        """Print the monitoring header"""
        print("="*80)
        print("PRIOR AUTHORIZATION AGENTS - REAL-TIME MONITORING")
        print("="*80)
        print(f"Started: {self.start_time.strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"System: {psutil.cpu_count()} CPUs, {self.format_memory(psutil.virtual_memory().total / 1024 / 1024)} RAM")
        print("-"*80)
    
    def print_agent_stats(self, stats: List[AgentStats]):
        """Print formatted agent statistics"""
        # Header
        print(f"{'AGENT':<25} {'STATUS':<8} {'PID':<8} {'CPU%':<8} {'MEMORY':<12} {'MEM%':<8} {'UPTIME':<10}")
        print("-"*80)
        
        # Agent stats
        for stat in stats:
            status_color = "🟢" if stat.status == "RUNNING" else "🔴" if stat.status == "STOPPED" else "🟡"
            pid_str = str(stat.pid) if stat.pid else "N/A"
            
            print(f"{stat.name:<23} {status_color} {stat.status:<6} {pid_str:<8} "
                  f"{stat.cpu_percent:>6.1f} {self.format_memory(stat.memory_mb):<12} "
                  f"{stat.memory_percent:>6.1f} {self.format_uptime(stat.uptime_seconds):<10}")
    
    def print_summary(self, stats: List[AgentStats]):
        """Print summary statistics"""
        running_count = sum(1 for s in stats if s.status == "RUNNING")
        total_cpu = sum(s.cpu_percent for s in stats)
        total_memory_mb = sum(s.memory_mb for s in stats)
        
        print("-"*80)
        print(f"SUMMARY: {running_count}/3 agents running | "
              f"Total CPU: {total_cpu:.1f}% | "
              f"Total Memory: {self.format_memory(total_memory_mb)}")
        print(f"Refresh Rate: {self.refresh_interval}s | Press Ctrl+C to exit")
        print("="*80)
    
    def save_stats_to_file(self, stats: List[AgentStats], filename: str = "agent_stats.json"):
        """Save current statistics to a JSON file"""
        stats_data = {
            "timestamp": datetime.now().isoformat(),
            "agents": []
        }
        
        for stat in stats:
            stats_data["agents"].append({
                "name": stat.name,
                "pid": stat.pid,
                "port": stat.port,
                "cpu_percent": stat.cpu_percent,
                "memory_mb": stat.memory_mb,
                "memory_percent": stat.memory_percent,
                "status": stat.status,
                "uptime_seconds": stat.uptime_seconds,
                "start_time": stat.start_time.isoformat() if stat.start_time else None
            })
        
        with open(filename, 'w') as f:
            json.dump(stats_data, f, indent=2)
    
    def run_monitoring(self, save_to_file: bool = False, log_file: str = None):
        """Run the monitoring loop"""
        try:
            iteration = 0
            while True:
                # Clear screen and print header
                self.clear_screen()
                self.print_header()
                
                # Find agent processes
                agent_processes = self.find_agent_processes()
                
                # Get statistics for each agent
                stats = []
                for port in sorted(self.agents_config.keys()):
                    process = agent_processes.get(port)
                    stat = self.get_agent_stats(port, process)
                    stats.append(stat)
                
                # Display statistics
                self.print_agent_stats(stats)
                self.print_summary(stats)
                
                # Save to file if requested
                if save_to_file:
                    self.save_stats_to_file(stats, log_file or f"agent_stats_{datetime.now().strftime('%Y%m%d')}.json")
                
                # Add to history for trend analysis
                self.stats_history.append({
                    "timestamp": datetime.now(),
                    "stats": stats.copy()
                })
                
                # Keep only last 100 entries
                if len(self.stats_history) > 100:
                    self.stats_history = self.stats_history[-100:]
                
                # Wait for next refresh
                time.sleep(self.refresh_interval)
                iteration += 1
                
        except KeyboardInterrupt:
            print("\n\nMonitoring stopped by user.")
            if save_to_file and self.stats_history:
                print("Final stats saved to file.")
        except Exception as e:
            print(f"\nError during monitoring: {e}")
            sys.exit(1)

def main():
    """Main function"""
    parser = argparse.ArgumentParser(description="Monitor Prior Authorization Agents")
    parser.add_argument("--refresh", "-r", type=float, default=2.0,
                       help="Refresh interval in seconds (default: 2.0)")
    parser.add_argument("--save", "-s", action="store_true",
                       help="Save statistics to JSON file")
    parser.add_argument("--log-file", "-f", type=str,
                       help="Log file name (default: agent_stats_YYYYMMDD.json)")
    parser.add_argument("--wait", "-w", type=int, default=10,
                       help="Wait time in seconds before starting monitoring (default: 10)")
    
    args = parser.parse_args()
    
    print("🚀 Starting Agent Monitor...")
    print(f"Waiting {args.wait} seconds for agents to start up...")
    
    # Wait for agents to start
    for i in range(args.wait, 0, -1):
        print(f"\rStarting monitoring in {i} seconds...", end="", flush=True)
        time.sleep(1)
    
    print("\n")
    
    # Create and run monitor
    monitor = AgentMonitor(refresh_interval=args.refresh)
    monitor.run_monitoring(save_to_file=args.save, log_file=args.log_file)

if __name__ == "__main__":
    main()