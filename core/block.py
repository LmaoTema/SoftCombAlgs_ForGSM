class Block:

    def __init__(self, is_working=True):
        self.is_working = is_working

    def process(self, *args):
        if not self.is_working:
            return args[0]


        return self._process(*args)
    
    def _process(self, *args):
        raise NotImplementedError("Block must implement _process")