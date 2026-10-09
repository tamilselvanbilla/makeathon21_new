"""Entry point: python src/main.py [--text] [--no-tts] [--offline]"""

import sys

from companion.app import main

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\nGoodbye.")
        sys.exit(0)
