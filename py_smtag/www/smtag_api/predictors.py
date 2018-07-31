import subprocess
import abc
import logging

class PredictorImplementor(metaclass=abc.ABCMeta):
    @abc.abstractmethod
    def complete(self, text, format, tag):
        pass
    @abc.abstractmethod
    def entity(self, text, format, tag):
        pass
    @abc.abstractmethod
    def role(self, text, format, tag):
        pass
    @abc.abstractmethod
    def tagger(self, text, format, tag):
        pass
    @abc.abstractmethod
    def panelize(self, text, format, tag):
        pass


class PythonPredictor(PredictorImplementor):
    def complete(self, text, format, tag):
        return "implementation pending"
    def entity(self, text, format, tag):
        return "implementation pending"
    def role(self, text, format, tag):
        return "implementation pending"
    def tagger(self, text, format, tag):
        return "implementation pending"
    def panelize(self, text, format, tag):
        return "implementation pending"

import re
def cleanup(text):
    text = re.sub('[\r\n\t]', ' ', text)
    text = re.sub(' +', ' ', text)
    return text

class LuaCliPredictor(PredictorImplementor):
    def __init__(self, smtag_lua_cli_path="../../sd-smtag", torch_path="th"):
        # path_to_bin = '/Users/alejandroriera/dev/sd-smtag/'
        # self.path_to_bin = path_to_bin
        self.smtag_lua_cli_path = smtag_lua_cli_path
        self.torch_path = torch_path
    def complete(self, text, format, tag):
        return self._run_command('smtag', text, format, tag)
    def entity(self, text, format, tag):
        return self._run_command('entity', text, format, tag)
    def role(self, text, format, tag):
        return self._run_command('role', text, format, tag)
    def tagger(self, text, format, tag):
        return self._run_command('tag', text, format, tag)
    def panelize(self, text, format, tag):
        return self._run_command('panelize', text, format, tag)
    def _run_command(self, method, text, format, tag):
        logger = logging.getLogger('smtag_api')
        logger.debug(f"LuaCliPredictor.{method}(format={format}, tag={tag})")
        text = cleanup(text)
        command = f'cd {self.smtag_lua_cli_path} && {self.torch_path} smtagCLI.lua -t "{text}" -D true -f {format} -m {method}'
        if len(tag) > 0:
            command = f"{command} -g {tag}"
        # print(command)
        result = subprocess.run([command], stdout=subprocess.PIPE, shell=True)
        # result = subprocess.run('cd /Users/alejandroriera/dev/sd-smtag/ && th smtagCli.lua -t "hallo" -D true -f xml -m smtag', stdout=subprocess.PIPE, shell=True).stdout.decode('utf-8')
        return result.stdout.decode('utf-8')

# if __name__ == 'main':
#     print("hi")
# import subprocess
# subprocess.run(['cd /Users/alejandroriera/dev/sd-smtag/ &&','th', 'smtagCli.lua', '-t "hallo"', '-D true', '-f xml', '-m smtag'], stdout=subprocess.PIPE).stdout.decode('utf-8')
# subprocess.run('cd /Users/alejandroriera/dev/sd-smtag/ && th smtagCli.lua -t "hallo" -D true -f xml -m smtag', stdout=subprocess.PIPE, shell=True).stdout.decode('utf-8')
# res = subprocess.check_output('cd /Users/alejandroriera/dev/sd-smtag/ && th smtagCli.lua -t "hallo" -D true -f xml -m smtag', stderr=subprocess.STDOUT)
# res = subprocess.check_output('ls -la', stderr=subprocess.STDOUT)

