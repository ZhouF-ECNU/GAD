def singleton(cls):
    _instance = {}

    def inner():
        if cls not in _instance:
            _instance[cls] = cls()
        return _instance[cls]
    return inner
    
@singleton
class writer2txt(object):

    def __init__(self) -> None:
        self.parameter ={}
        pass

    def set_pic_save_path(self, set_pic_save_path):
        self.set_pic_save_path = set_pic_save_path

    def set_output_name(self, output_name):
        self.output_name = output_name

    def set_file_save_path(self, file_save_path):
        self.file_save_path = file_save_path

    def set_path(self, path, log_path = None) -> None:
        self.path = path
        self.log_path = log_path

    def write_txt(self, content):
        with open(self.path, 'a+') as f:
            f.write(str(content))
            f.write('\n')

    def log(self, content):
        if self.log_path is None:
            return
        with open(self.log_path, 'a+') as f:
            f.write(str(content))
            f.write('\n')
    
    def add_para(self, key, value):
        self.parameter[key] = value
    
    def get_para(self, key):
        return self.parameter[key]