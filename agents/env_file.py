__all__ = ['EnvFile', 'get_env', 'load_env', 'parse_env']


import os


def parse_env(line: str) -> dict:
    """parse line and return a dictionary with variable value"""
    if line.lstrip().startswith('#'):
        return {}
    if not line.lstrip():
        return {}
    """find the second occurrence of a quote mark:"""
    if line.find("export ") == 0:
        line = line.replace("export ", "", 1)
    quote_delimit = max(line.find("'", line.find("'") + 1), line.find('"', line.rfind('"')) + 1)
    """find first comment mark after second quote mark"""
    if '#' in line and not(line.strip().startswith('"') and line.strip().endswith('"')):
        line = line[:line.find('#', quote_delimit)]
    key, value = map(
        lambda x: x.strip().strip('\'').strip('"'),
        line.split('=', 1)
    )
    return {key: value}


class EnvFile(dict):
    """.env file class"""
    path = None

    def __init__(self, path):
        super().__init__()
        if not os.path.exists(path):
            raise OSError("%s DOESN'T EXIST!" % os.path.abspath(path))
        self.path = os.path.abspath(os.path.expanduser(path))

    def load(self, **kwargs):
        for line in open(self.path).read().splitlines():
            parsed_env = parse_env(line)
            self.update(parsed_env)
        for k, v in kwargs.items():
            self[k] = v
        return self

    def __setitem__(self, key, value):
        super(EnvFile, self).__setitem__(key, value)

    def __delitem__(self, key):
        super(EnvFile, self).__delitem__(key)


def get_env(path=".env", **kwargs) -> EnvFile:
    """return a dictionary wit .env file variables"""
    if not path:
        path = ".env"
    envfile = EnvFile(path)
    return envfile.load(**kwargs)


def load_env(path=".env", **kwargs):
    """set environment variables from .env file"""
    path = ".env" if not path else os.path.abspath(os.path.expanduser(path))
    env = get_env(path, **kwargs)
    os.environ.update(env)
    return env
