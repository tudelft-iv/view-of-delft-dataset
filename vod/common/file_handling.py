from os.path import exists
import os


def get_frame_list(path: str) -> list:
    """
This function reads a textfile that contains the frame numbers.
    :param path: The path to the textfile that contains the frames.
    :return: A list with the frame numbers.
    """

    if exists(path):
        with open(path, 'r') as file:
            lines = file.readlines()
            lines = [text_line.rstrip() for text_line in lines]

        return lines

    else:
        raise ValueError(f"{path} does not exist!")


def get_frame_list_from_folder(path):
    """
This function reads a folder and returns a list with the frame numbers.
    :param path: Directory path
    :return: lost with frames
    """
    picture_list = []
    for root, dirs, files in os.walk(path, topdown=False):
        for name in files:
            if name.endswith('.txt'):
                picture_list.append(name[:-4])
    return sorted(picture_list)
