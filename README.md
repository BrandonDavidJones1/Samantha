Running the Bot from Source Code (Recommended for macOS & Linux Users)
This method involves running the Python script directly. It's recommended if you're on macOS or Linux, or if you prefer not to use a pre-compiled executable.
Prerequisites:
Git: You'll need Git installed.
macOS: Open Terminal and type git --version. If not installed, it might prompt you to install Xcode Command Line Tools (which includes Git).
Linux: Use your distribution's package manager (e.g., sudo apt install git or sudo yum install git).
Windows: Download from git-scm.com.
Python 3: Version 3.8 or higher is recommended.
macOS/Linux: Check with python3 --version. Install from your package manager or python.org if needed.
Windows: Download from python.org. Ensure Python is added to your PATH during installation.
Pip: Python's package installer (usually comes with Python 3).
One-Time Setup:
Open your Terminal (macOS/Linux) or Command Prompt/PowerShell (Windows).
Create a dedicated folder for the bot and navigate into it:
mkdir SamanthaBot
cd SamanthaBot
Use code with caution.
Bash
Clone the repository:
git clone https://github.com/BrandonDavidJones1/Samantha.git .
Use code with caution.
Bash
(Note the space and dot . at the end to clone into the current directory).
Create and activate a Python virtual environment (highly recommended):
macOS/Linux:
python3 -m venv venv
source venv/bin/activate
Use code with caution.
Bash
Windows (Command Prompt):
python -m venv venv
venv\Scripts\activate.bat
Use code with caution.
Bash
Windows (PowerShell):
python -m venv venv
.\venv\Scripts\Activate.ps1
Use code with caution.
Bash
(If you get an error about script execution policy in PowerShell, you might need to run: Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser first, then try activating again).
You should see (venv) at the beginning of your terminal/command prompt.
Install required Python libraries:
(Ensure your virtual environment is active)
pip install -r requirements.txt
Use code with caution.
Bash
Create the Environment File (.env):
In the main SamanthaBot project folder (the one containing bot.py), create a new file named exactly .env (starting with a dot).
Open this .env file with a plain text editor (e.g., Notepad on Windows, TextEdit on Mac in "Plain Text" mode, VS Code).
Add the following line, replacing YOUR_DISCORD_BOT_TOKEN_HERE with the actual Discord bot token:
DISCORD_TOKEN=YOUR_DISCORD_BOT_TOKEN_HERE
Use code with caution.
Save and close the .env file. Never commit this file to GitHub if it contains your actual token. The .gitignore file in this repository should prevent this.
Running the Bot:
Open your Terminal/Command Prompt.
Navigate to the bot's project directory (e.g., cd path/to/your/SamanthaBot).
Activate the virtual environment (if not already active):
macOS/Linux: source venv/bin/activate
Windows: venv\Scripts\activate.bat (or .\venv\Scripts\Activate.ps1 for PowerShell)
Run the bot script:
macOS/Linux: python3 bot.py
Windows: python bot.py
The bot will start, and you should see log messages in your terminal indicating it has connected to Discord.
To stop the bot: Go to the terminal window where it's running and press Ctrl + C.
Getting Updates:
When new features or fixes are available in the GitHub repository:
Open your Terminal/Command Prompt.
Navigate to the bot's project directory.
Activate the virtual environment.
Pull the latest changes from GitHub:
git pull
Use code with caution.
Bash
Update dependencies (if there were changes to requirements.txt):
pip install -r requirements.txt
Use code with caution.
Bash
Run the bot as described in the "Running the Bot" section above.