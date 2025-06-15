#!/usr/bin/env python3
"""
==============================================================================
                    DESKTOP GUI ASSISTANT WITH VOICE COMMANDS
==============================================================================

A complete terminal-like desktop assistant with voice recognition and TTS.

FILE STRUCTURE (Single .py file):
├── Imports & Dependencies
├── VoiceAssistant Class
│   ├── __init__() - Initialize application
│   ├── GUI Setup Methods
│   │   ├── setup_window() - Configure main window
│   │   ├── setup_widgets() - Create UI components
│   │   └── setup_voice_components() - Initialize speech systems
│   ├── Communication Methods
│   │   ├── setup_message_queue() - Thread-safe messaging
│   │   ├── process_queue() - Handle background updates
│   │   ├── add_terminal_message() - Display messages
│   │   └── speak_text() - Text-to-speech functions
│   ├── Input Handlers
│   │   ├── on_text_command() - Handle typed commands
│   │   ├── on_voice_command() - Handle voice button
│   │   └── listen_for_voice() - Voice recognition thread
│   ├── Command Processor
│   │   ├── process_command() - Main command router
│   │   └── execute_command() - Command implementations
│   └── System Operations
│       ├── File operations (read, list, navigate)
│       ├── System commands (notepad, calculator, etc.)
│       └── Utility functions (help, system info)
└── main() - Application entry point

REQUIRED PACKAGES:
pip install speechrecognition pyttsx3 pyaudio

AUTHOR: AvijnanPurkait
==============================================================================
"""

# ============================= IMPORTS =====================================
import tkinter as tk
from tkinter import ttk, scrolledtext, messagebox, filedialog
import threading
import queue
import os
import subprocess
import platform
import datetime
import speech_recognition as sr
import pyttsx3
import time


# ============================ MAIN CLASS ===================================
class VoiceAssistant:
    """
    Main application class for the Voice Assistant GUI
    
    Architecture:
    - GUI runs on main thread
    - Voice recognition runs on background threads
    - TTS runs on separate threads to avoid blocking
    - Message queue ensures thread-safe GUI updates
    """
    
    # ========================= INITIALIZATION ===============================
    def __init__(self, root):
        """Initialize the Voice Assistant application"""
        self.root = root
        self.setup_window()
        self.setup_widgets()
        self.setup_voice_components()
        self.setup_message_queue()
        
        # Application state
        self.listening = False
        self.current_directory = os.getcwd()
        
        # Start the GUI message processor
        self.process_queue()
        
        # Welcome message
        self.add_terminal_message("SYSTEM", "Desktop Assistant initialized. Type 'help' for commands.")
        self.speak_text("Welcome to your desktop assistant. I'm ready to help!")
    
    # ========================== GUI SETUP ===================================
    
    def setup_window(self):
        """Configure the main window with accessible design"""
        self.root.title("Desktop Assistant Terminal")
        self.root.geometry("800x600")
        self.root.configure(bg='#1a1a1a')  # Dark background for high contrast
        
        # Make window resizable
        self.root.rowconfigure(0, weight=1)
        self.root.columnconfigure(0, weight=1)
        
        # Set minimum size
        self.root.minsize(600, 400)
    
    def setup_widgets(self):
        """Create and arrange GUI widgets"""
        # Main frame
        main_frame = ttk.Frame(self.root)
        main_frame.grid(row=0, column=0, sticky="nsew", padx=10, pady=10)
        main_frame.rowconfigure(0, weight=1)
        main_frame.columnconfigure(0, weight=1)
        
        # Configure style for accessibility
        style = ttk.Style()
        style.theme_use('clam')
        
        # Terminal display area (scrolled text widget)
        self.terminal_display = scrolledtext.ScrolledText(
            main_frame,
            wrap=tk.WORD,
            width=80,
            height=25,
            bg='#000000',      # Black background
            fg='#00ff00',      # Green text (classic terminal look)
            font=('Consolas', 11),  # Monospace font for readability
            insertbackground='#00ff00',  # Green cursor
            selectbackground='#333333',
            state=tk.DISABLED  # Read-only initially
        )
        self.terminal_display.grid(row=0, column=0, sticky="nsew", pady=(0, 10))
        
        # Input frame
        input_frame = ttk.Frame(main_frame)
        input_frame.grid(row=1, column=0, sticky="ew")
        input_frame.columnconfigure(0, weight=1)
        
        # Command input
        self.command_var = tk.StringVar()
        self.command_entry = ttk.Entry(
            input_frame,
            textvariable=self.command_var,
            font=('Consolas', 11),
            width=60
        )
        self.command_entry.grid(row=0, column=0, sticky="ew", padx=(0, 10))
        self.command_entry.bind('<Return>', self.on_text_command)
        self.command_entry.focus()
        
        # Voice button
        self.voice_button = ttk.Button(
            input_frame,
            text="🎤 Voice",
            command=self.on_voice_command,
            width=12
        )
        self.voice_button.grid(row=0, column=1)
        
        # Status label
        self.status_var = tk.StringVar(value="Ready")
        self.status_label = ttk.Label(
            main_frame,
            textvariable=self.status_var,
            font=('Arial', 9),
            foreground='#666666'
        )
        self.status_label.grid(row=2, column=0, sticky="w", pady=(5, 0))
    
    # ========================= VOICE SETUP ==================================
    
    def setup_voice_components(self):
        """Initialize speech recognition and text-to-speech"""
        try:
            # Initialize speech recognition
            self.recognizer = sr.Recognizer()
            self.microphone = sr.Microphone()
            
            # Adjust for ambient noise
            with self.microphone as source:
                self.recognizer.adjust_for_ambient_noise(source, duration=1)
            
            # Initialize text-to-speech
            self.tts_engine = pyttsx3.init()
            
            # Configure TTS settings
            voices = self.tts_engine.getProperty('voices')
            if voices:
                # Try to use a clearer voice if available
                for voice in voices:
                    if 'female' in voice.name.lower() or 'zira' in voice.name.lower():
                        self.tts_engine.setProperty('voice', voice.id)
                        break
            
            # Set speech rate and volume
            self.tts_engine.setProperty('rate', 180)  # Slower for clarity
            self.tts_engine.setProperty('volume', 0.8)
            
        except Exception as e:
            self.add_terminal_message("ERROR", f"Voice components initialization failed: {str(e)}")
    
    # ====================== COMMUNICATION METHODS ===========================
    
    def setup_message_queue(self):
        """Set up thread-safe message queue for GUI updates"""
        self.message_queue = queue.Queue()
    
    def process_queue(self):
        """Process messages from background threads"""
        try:
            while True:
                msg_type, message = self.message_queue.get_nowait()
                if msg_type == "terminal":
                    speaker, text = message
                    self.add_terminal_message(speaker, text)
                elif msg_type == "status":
                    self.status_var.set(message)
                elif msg_type == "speak":
                    self.speak_text_async(message)
        except queue.Empty:
            pass
        
        # Schedule next check
        self.root.after(100, self.process_queue)
    
    def add_terminal_message(self, speaker, message):
        """Add a message to the terminal display"""
        timestamp = datetime.datetime.now().strftime("%H:%M:%S")
        
        # Enable editing temporarily
        self.terminal_display.config(state=tk.NORMAL)
        
        # Color coding for different speakers
        if speaker == "USER":
            color = '#00ffff'  # Cyan for user
            prefix = f"[{timestamp}] USER> "
        elif speaker == "SYSTEM":
            color = '#ffff00'  # Yellow for system
            prefix = f"[{timestamp}] SYSTEM> "
        elif speaker == "ERROR":
            color = '#ff0000'  # Red for errors
            prefix = f"[{timestamp}] ERROR> "
        else:
            color = '#00ff00'  # Green for assistant
            prefix = f"[{timestamp}] ASSISTANT> "
        
        # Insert the message
        self.terminal_display.insert(tk.END, prefix)
        self.terminal_display.insert(tk.END, f"{message}\n\n")
        
        # Auto-scroll to bottom
        self.terminal_display.see(tk.END)
        
        # Disable editing
        self.terminal_display.config(state=tk.DISABLED)
        
        # Update the display
        self.root.update_idletasks()
    
    def speak_text(self, text):
        """Convert text to speech (blocking)"""
        try:
            self.tts_engine.say(text)
            self.tts_engine.runAndWait()
        except Exception as e:
            print(f"TTS Error: {e}")
    
    def speak_text_async(self, text):
        """Convert text to speech in a separate thread"""
        def speak_worker():
            self.speak_text(text)
        
        thread = threading.Thread(target=speak_worker, daemon=True)
        thread.start()
    
    # ======================== INPUT HANDLERS ================================
    
    def on_text_command(self, event=None):
        """Handle text command input"""
        command = self.command_var.get().strip()
        if command:
            self.command_var.set("")  # Clear input
            self.process_command(command, "text")
    
    def on_voice_command(self):
        """Handle voice command button click"""
        if not self.listening:
            thread = threading.Thread(target=self.listen_for_voice, daemon=True)
            thread.start()
    
    def listen_for_voice(self):
        """Listen for voice input in a separate thread"""
        self.listening = True
        self.message_queue.put(("status", "Listening... Speak now!"))
        self.voice_button.config(state="disabled")
        
        try:
            with self.microphone as source:
                # Listen for audio
                audio = self.recognizer.listen(source, timeout=5, phrase_time_limit=10)
            
            self.message_queue.put(("status", "Processing speech..."))
            
            # Recognize speech
            try:
                command = self.recognizer.recognize_google(audio)
                self.message_queue.put(("terminal", ("USER", f"[Voice] {command}")))
                self.process_command(command, "voice")
            except sr.UnknownValueError:
                self.message_queue.put(("terminal", ("ERROR", "Could not understand audio")))
                self.message_queue.put(("speak", "Sorry, I couldn't understand what you said."))
            except sr.RequestError as e:
                self.message_queue.put(("terminal", ("ERROR", f"Speech recognition error: {e}")))
                self.message_queue.put(("speak", "Speech recognition service is unavailable."))
        
        except sr.WaitTimeoutError:
            self.message_queue.put(("terminal", ("SYSTEM", "Voice input timeout")))
        except Exception as e:
            self.message_queue.put(("terminal", ("ERROR", f"Voice input error: {str(e)}")))
        
        finally:
            self.listening = False
            self.message_queue.put(("status", "Ready"))
            self.root.after(100, lambda: self.voice_button.config(state="normal"))
    
    # ======================= COMMAND PROCESSOR ==============================
    
    def process_command(self, command, input_type):
        """Process and execute commands"""
        command_lower = command.lower().strip()
        
        # Add user command to terminal
        if input_type == "text":
            self.add_terminal_message("USER", command)
        
        try:
            response = self.execute_command(command_lower)
            self.add_terminal_message("ASSISTANT", response)
            self.speak_text_async(response)
            
        except Exception as e:
            error_msg = f"Command execution failed: {str(e)}"
            self.add_terminal_message("ERROR", error_msg)
            self.speak_text_async("Sorry, there was an error executing that command.")
    
    def execute_command(self, command):
        """Execute specific commands and return response"""
        # Help command
        if command in ['help', 'commands']:
            return self.get_help_text()
        
        # Time and date commands
        elif command in ['time', 'what time is it']:
            return f"Current time: {datetime.datetime.now().strftime('%I:%M %p')}"
        
        elif command in ['date', 'what date is it', 'today']:
            return f"Today's date: {datetime.datetime.now().strftime('%A, %B %d, %Y')}"
        
        # Directory commands
        elif command in ['dir', 'ls', 'list directory']:
            return self.list_directory()
        
        elif command.startswith('cd '):
            path = command[3:].strip()
            return self.change_directory(path)
        
        elif command in ['pwd', 'current directory']:
            return f"Current directory: {self.current_directory}"
        
        # File operations
        elif command.startswith('read ') or command.startswith('read file '):
            filename = command.replace('read file ', '').replace('read ', '').strip()
            return self.read_file_aloud(filename)
        
        elif command.startswith('open '):
            filename = command[5:].strip()
            return self.open_file(filename)
        
        # System commands
        elif command in ['open notepad', 'notepad']:
            return self.open_notepad()
        
        elif command in ['open calculator', 'calculator', 'calc']:
            return self.open_calculator()
        
        elif command.startswith('search '):
            query = command[7:].strip()
            return self.web_search(query)
        
        # System information
        elif command in ['system info', 'system information']:
            return self.get_system_info()
        
        # Exit commands
        elif command in ['exit', 'quit', 'close', 'goodbye']:
            self.speak_text("Goodbye! Have a great day!")
            self.root.after(2000, self.root.quit)  # Exit after 2 seconds
            return "Goodbye! Closing application..."
        
        else:
            return f"Unknown command: '{command}'. Type 'help' for available commands."
    
    # ======================= SYSTEM OPERATIONS ==============================
    
    def get_help_text(self):
        """Return help text with available commands"""
        help_text = """Available Commands:
        
• help - Show this help message
• time - Get current time
• date - Get current date
• dir/ls - List directory contents
• cd [path] - Change directory
• pwd - Show current directory
• read [filename] - Read file content aloud
• open [filename] - Open file with default program
• notepad - Open Notepad
• calculator - Open Calculator
• search [query] - Open web search
• system info - Show system information
• exit/quit - Close application

You can use either text input or voice commands!"""
        return help_text
    
    def list_directory(self):
        """List contents of current directory"""
        try:
            items = os.listdir(self.current_directory)
            if not items:
                return "Directory is empty."
            
            dirs = [item for item in items if os.path.isdir(os.path.join(self.current_directory, item))]
            files = [item for item in items if os.path.isfile(os.path.join(self.current_directory, item))]
            
            result = f"Contents of {self.current_directory}:\n"
            if dirs:
                result += f"Directories ({len(dirs)}): {', '.join(dirs[:10])}"
                if len(dirs) > 10:
                    result += f" ... and {len(dirs) - 10} more"
                result += "\n"
            
            if files:
                result += f"Files ({len(files)}): {', '.join(files[:10])}"
                if len(files) > 10:
                    result += f" ... and {len(files) - 10} more"
            
            return result
        except Exception as e:
            return f"Error listing directory: {str(e)}"
    
    def change_directory(self, path):
        """Change current directory"""
        try:
            if path == "..":
                new_path = os.path.dirname(self.current_directory)
            elif os.path.isabs(path):
                new_path = path
            else:
                new_path = os.path.join(self.current_directory, path)
            
            if os.path.exists(new_path) and os.path.isdir(new_path):
                self.current_directory = os.path.abspath(new_path)
                return f"Changed directory to: {self.current_directory}"
            else:
                return f"Directory not found: {path}"
        except Exception as e:
            return f"Error changing directory: {str(e)}"
    
    def read_file_aloud(self, filename):
        """Read a text file and speak its contents"""
        try:
            # Handle relative paths
            if not os.path.isabs(filename):
                filepath = os.path.join(self.current_directory, filename)
            else:
                filepath = filename
            
            if not os.path.exists(filepath):
                return f"File not found: {filename}"
            
            with open(filepath, 'r', encoding='utf-8') as file:
                content = file.read()
            
            if len(content) > 1000:
                content = content[:1000] + "... (truncated for speech)"
            
            # Speak the content in a separate thread
            def speak_file_content():
                self.speak_text(f"Reading file {filename}. {content}")
            
            thread = threading.Thread(target=speak_file_content, daemon=True)
            thread.start()
            
            return f"Reading file: {filename}\n{content[:500]}{'...' if len(content) > 500 else ''}"
        
        except Exception as e:
            return f"Error reading file: {str(e)}"
    
    def open_file(self, filename):
        """Open a file with the default system program"""
        try:
            if not os.path.isabs(filename):
                filepath = os.path.join(self.current_directory, filename)
            else:
                filepath = filename
            
            if not os.path.exists(filepath):
                return f"File not found: {filename}"
            
            system = platform.system()
            if system == "Windows":
                os.startfile(filepath)
            elif system == "Darwin":  # macOS
                subprocess.run(["open", filepath])
            else:  # Linux
                subprocess.run(["xdg-open", filepath])
            
            return f"Opened file: {filename}"
        
        except Exception as e:
            return f"Error opening file: {str(e)}"
    
    def open_notepad(self):
        """Open Notepad or default text editor"""
        try:
            system = platform.system()
            if system == "Windows":
                subprocess.Popen(["notepad.exe"])
                return "Opened Notepad"
            elif system == "Darwin":  # macOS
                subprocess.Popen(["open", "-a", "TextEdit"])
                return "Opened TextEdit"
            else:  # Linux
                subprocess.Popen(["gedit"])
                return "Opened text editor"
        except Exception as e:
            return f"Error opening text editor: {str(e)}"
    
    def open_calculator(self):
        """Open system calculator"""
        try:
            system = platform.system()
            if system == "Windows":
                subprocess.Popen(["calc.exe"])
                return "Opened Calculator"
            elif system == "Darwin":  # macOS
                subprocess.Popen(["open", "-a", "Calculator"])
                return "Opened Calculator"
            else:  # Linux
                subprocess.Popen(["gnome-calculator"])
                return "Opened Calculator"
        except Exception as e:
            return f"Error opening calculator: {str(e)}"
    
    def web_search(self, query):
        """Open web browser with search query"""
        try:
            import webbrowser
            search_url = f"https://www.google.com/search?q={query.replace(' ', '+')}"
            webbrowser.open(search_url)
            return f"Opened web search for: {query}"
        except Exception as e:
            return f"Error opening web search: {str(e)}"
    
    def get_system_info(self):
        """Get basic system information"""
        try:
            system = platform.system()
            release = platform.release()
            machine = platform.machine()
            processor = platform.processor()
            
            info = f"System: {system} {release}\n"
            info += f"Architecture: {machine}\n"
            info += f"Processor: {processor}\n"
            info += f"Python Version: {platform.python_version()}"
            
            return info
        except Exception as e:
            return f"Error getting system info: {str(e)}"


# ========================== APPLICATION ENTRY ==============================
def main():
    """
    Main function to run the application
    
    Checks for required packages and initializes the GUI application.
    Handles graceful error reporting if dependencies are missing.
    """
    # Check for required packages
    try:
        import speech_recognition
        import pyttsx3
    except ImportError as e:
        print(f"Missing required package: {e}")
        print("Please install required packages:")
        print("pip install speechrecognition pyttsx3 pyaudio")
        return
    
    # Create and run the application
    root = tk.Tk()
    
    try:
        app = VoiceAssistant(root)
        root.mainloop()
    except KeyboardInterrupt:
        print("\nApplication interrupted by user")
    except Exception as e:
        print(f"Application error: {e}")
        messagebox.showerror("Error", f"Application error: {str(e)}")


if __name__ == "__main__":
    main()