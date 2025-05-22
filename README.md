# Running the Bot from Source Code

This method is recommended for users on macOS, Linux, or those who prefer running Python scripts directly.

## Prerequisites

Before you begin, ensure you have the following installed on your system:

*   **Git:**
    *   To check if Git is installed, open your terminal and type:
        ```bash
        git --version
        ```
    *   *macOS:* If not installed, typing `git` in the terminal might prompt you to install Xcode Command Line Tools (which includes Git).
    *   *Linux:* Use your distribution's package manager (e.g., `sudo apt update && sudo apt install git` or `sudo yum install git`).
    *   *Windows:* Download and install from [git-scm.com](https://git-scm.com/).
*   **Python 3:** Version 3.8 or higher is recommended.
    *   To check, type `python3 --version` (or `python --version` on Windows if `python3` isn't recognized).
    *   Download from [python.org](https://www.python.org/) if needed. Ensure Python is added to your system's PATH during installation on Windows.
*   **Pip:** Python's package installer (usually included with Python 3).

## One-Time Setup Instructions

1.  **Open your Terminal** (on macOS/Linux) or **Command Prompt/PowerShell** (on Windows).

2.  **Create a dedicated folder** for the bot and navigate into it. Replace `SamanthaBot` with your preferred folder name if desired.
    ```bash
    mkdir SamanthaBot
    cd SamanthaBot
    ```

3.  **Clone the repository** into the current folder:
    ```bash
    git clone https://github.com/BrandonDavidJones1/Samantha.git .
    ```
    *(The `.` at the end ensures it clones into the current directory).*

4.  **Set up a Python virtual environment.** This is highly recommended to manage dependencies cleanly.
    *   Navigate into the cloned repository directory if you're not already there (it might have cloned into a subdirectory named `Samantha` if you omitted the `.` above. If so, `cd Samantha`).
    *   Create the virtual environment:
        ```bash
        python3 -m venv venv 
        ```
        *(On Windows, you might use `python` instead of `python3`)*.
    *   Activate the virtual environment:
        *   **macOS/Linux:**
            ```bash
            source venv/bin/activate
            ```
        *   **Windows (Command Prompt):**
            ```bash
            venv\Scripts\activate.bat
            ```
        *   **Windows (PowerShell):**
            ```bash
            .\venv\Scripts\Activate.ps1
            ```
            *(If PowerShell gives an error about script execution, you might need to run `Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser` first, then try activating again).*
        *   Your terminal prompt should now indicate that the virtual environment `(venv)` is active.

5.  **Install required Python libraries:**
    (Ensure your virtual environment `(venv)` is active)
    ```bash
    pip install -r requirements.txt
    ```

6.  **Create the Environment Configuration File (`.env`):**
    *   In the main project folder (where `bot.py` and `requirements.txt` are located), create a new file named exactly `.env` (it starts with a dot and has no extension).
    *   Open this `.env` file with a plain text editor (e.g., Notepad, VS Code, TextEdit in "Plain Text" mode).
    *   Add the following line, replacing `YOUR_DISCORD_BOT_TOKEN_HERE` with the actual Discord bot token:
        ```env
        DISCORD_TOKEN=YOUR_DISCORD_BOT_TOKEN_HERE
        ```
    *   Save and close the `.env` file.
    *   **Security Note:** This `.env` file contains your sensitive bot token and should **never** be committed to public Git repositories. The `.gitignore` file in this project is configured to prevent this.

## Running the Bot

1.  **Open your Terminal/Command Prompt.**
2.  **Navigate to the bot's project directory** (e.g., `cd path/to/your/SamanthaBot`).
3.  **Activate the virtual environment** (if it's not still active from setup):
    *   *macOS/Linux:* `source venv/bin/activate`
    *   *Windows:* `venv\Scripts\activate.bat` (or `.\venv\Scripts\Activate.ps1` for PowerShell)
4.  **Start the bot:**
    *   *macOS/Linux:*
        ```bash
        python3 bot.py
        ```
    *   *Windows:*
        ```bash
        python bot.py
        ```
5.  The bot will start running. You should see messages in your terminal indicating it has connected to Discord and is ready.
6.  **To keep the bot running, this terminal window must remain open.** You can minimize it.
7.  **To stop the bot:** Return to the terminal window where the bot is running and press `Ctrl + C`.

## Getting Updates

To update the bot with the latest changes from the GitHub repository:

1.  **Stop the bot** if it's currently running (`Ctrl + C` in its terminal window).
2.  **Open your Terminal/Command Prompt** and navigate to the bot's project directory.
3.  **Activate the virtual environment.**
4.  **Pull the latest code changes:**
    ```bash
    git pull
    ```
5.  **Update Python libraries** (if there were changes to `requirements.txt`):
    ```bash
    pip install -r requirements.txt
    ```
6.  **Run the bot** again using the instructions in the "Running the Bot" section above.