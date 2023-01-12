"""
This program is free software: you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation, either version 3 of the License, or
(at your option) any later version.

This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU General Public License for more details.

You should have received a copy of the GNU General Public License
along with this program.  If not, see <https://www.gnu.org/licenses/>.
"""

import os
from pathlib import Path


def getListOfFiles(dirName):
    # create a list of file and sub directories 
    # names in the given directory 
    listOfFile = os.listdir(dirName)
    allFiles = list()
    # Iterate over all the entries
    for entry in listOfFile:
        # Create full path
        fullPath = os.path.join(dirName, entry)
        # If entry is a directory then get the list of files in this directory 
        if os.path.isdir(fullPath):
            allFiles = allFiles + getListOfFiles(fullPath)
        else:
            allFiles.append(fullPath)

    return sorted(allFiles)


def remove_temp_files(directory='../data', key='/._'):
    """ Removes temp data files, marked with the ._ prefix which are created
    sometimes (at least by macOS) """
    files_removed = 0
    all_files = getListOfFiles(directory)

    for filename in all_files:
        if key in filename:
            os.remove(filename)
            files_removed += 1

    print(f'Removed {files_removed} files.')


cwd = Path(__file__).parent
os.chdir(cwd)

remove_temp_files(os.getcwd(), '._')
