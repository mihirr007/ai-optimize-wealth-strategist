import time
from typing import Callable, Optional
from colorama import Fore, Style, init

init(autoreset=True)


class ProgressTracker:
    def __init__(self):
        self.start_time = None
        self.handlers = []
        self.current_status = {}
    
    def start(self):
        """Start progress tracking"""
        self.start_time = time.time()
        self.current_status = {}
        print(f"{Fore.CYAN}Starting wealth management analysis...{Style.RESET_ALL}")
    
    def stop(self):
        """Stop progress tracking"""
        if self.start_time:
            elapsed = time.time() - self.start_time
            print(f"{Fore.GREEN}Analysis completed in {elapsed:.2f} seconds{Style.RESET_ALL}")
            self.start_time = None
    
    def update_status(self, agent_name: str, client_id: Optional[str], status: str):
        """Update status for an agent"""
        key = f"{agent_name}_{client_id}" if client_id else agent_name
        self.current_status[key] = {
            "agent": agent_name,
            "client_id": client_id,
            "status": status,
            "timestamp": time.time()
        }
        
        # Print status update
        client_display = f" [{client_id}]" if client_id else ""
        print(f"{Fore.YELLOW}✓ {agent_name}{client_display} {status}{Style.RESET_ALL}")
        
        # Notify handlers
        for handler in self.handlers:
            try:
                handler(agent_name, client_id, status, None, time.time())
            except Exception as e:
                print(f"Error in progress handler: {e}")
    
    def register_handler(self, handler: Callable):
        """Register a progress update handler"""
        self.handlers.append(handler)
    
    def get_current_status(self) -> dict:
        """Get current status of all agents"""
        return self.current_status.copy()


# Global progress tracker instance
progress = ProgressTracker() 