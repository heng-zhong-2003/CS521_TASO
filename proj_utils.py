from colorama import Fore, Style
import inspect
from typing import NoReturn


def todo(message: str = '') -> NoReturn:
    frame = inspect.stack()[1]
    filename = frame.filename
    lineno = frame.lineno
    raise NotImplementedError(
        f'{Fore.RED}Not yet implemented at {filename}:{lineno}, {message}{Style.RESET_ALL}')
