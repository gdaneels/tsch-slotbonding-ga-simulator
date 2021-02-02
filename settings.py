import json

class Settings:
    def __init__(self, file_path):
        self.settings_file = file_path
        with open(file_path) as json_file:
            self.settings = json.load(json_file)

    def __getitem__(self, key):
        return self.settings[key]


