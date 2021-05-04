import os

class Assessment:

    def __init__(self, library, assignments=None):
        self.library = library
        self.assignments = assignments
        self.real_data = {}

    def breakdown(self):
        if self.assignments == None:
            self.library_sort()
        else:
            self.mixed_sort()

    def library_sort(self):
        with open(f'{self.library}', 'r') as lib_file:
            readstrings = []
            id = ''
            for line in lib_file.readline():
                if line.startswith('>'):
                    if id in self.real_data.keys():
                        self.real_data[id] = [type, ''.join(readstrings)]
                    id, type = line.split(' ')
                    id = str(id[1:])
                else:
                    readstrings.append(line)

    def mixed_sort(self):
        with open(f'{self.assignments}', 'r') as info_file:
            for line in info_file.readline():
                id, type = line.split(' ')
                self.real_data[id] = [type]

        with open(f'{self.library}', 'r') as lib_file:
            readstrings = []
            id = ''
            for line in lib_file.readline():
                if line.startswith('>'):
                    if id in self.real_data.keys():
                        self.real_data[id].append(''.join(readstrings))
                    id, type = line.split(' ')
                    id = str(id[1:])
                else:
                    readstrings.append(line)

