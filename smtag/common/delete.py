import os
import argparse
from .progress import progress
from .. import config

def delete(path, extension):
    dir_path = os.path.join(config.working_directory, path)
    file_list = [f for f in os.listdir(dir_path) if os.path.splitext(f)[-1] == extension]
    not_removed = []
    N = len(file_list)
    confirm = input("Are you sure you want to delete {N} files from {dir_path}? [Y/n]: ")
    if confirm == "Y":
        for i, f in enumerate(file_list):
            progress(i, N, f"deleting {f}                                       ")
            try:
                os.remove(os.path.join(dir_path, f))
            except Exception as e:
                not_removed.append((f, e))
        if not_removed:
            for f, e in not_removed:
                print(f, e)
    print(f"\n{N} files deleted.")

def main():
    parser = config.create_argument_parser_with_defaults(description='Exracting visual context vectors from images')
    parser.add_argument('--ext', help='Extension of the files to be deleted.')
    parser.add_argument('path', help='Path to the directory containing the files to delete.')
    args = parser.parse_args()
    extension = args.ext
    path = args.path
    if extension and path:
        delete(args.path, args.ext)
    else:
        print("specify an extension")

if __name__ == "__main__":
    main()