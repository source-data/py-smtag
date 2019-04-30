# https://cognitiveintegratorapp.azurewebsites.net/integrator/dashboard
# install gcloud for mac https://cloud.google.com/sdk/docs/quickstart-macos
# of linux https://cloud.google.com/sdk/docs/quickstart-linux
# see also https://googleapis.github.io/google-cloud-python/latest/vision/index.html
# https://googleapis.github.io/google-cloud-python/latest/vision/
# https://cloud.google.com/vision/docs/ocr
#
# gcloud components update &&
# gcloud components install beta
# ./bin/gcloud init
# # * Commands that require authentication will use thomas.lemberger@gmail.com by default
# * Commands will reference project `smarttag-2` by default
# Run `gcloud help config` to learn how to change individual settings
# add google-cloud-vision to requirements
# pip install -r requirements
# https://cloud.google.com/vision/docs/libraries#client-libraries-install-python
# https://cloud.google.com/vision/docs/quickstart-client-libraries
# export GOOGLE_APPLICATION_CREDENTIALS="[PATH]"
# export GOOGLE_APPLICATION_CREDENTIALS="/Users/lemberger/Documents/code/cloud/smarttag-2-5b02d5e85409.json"
# google-auth
# or use credentials=credentials (google.auth.credentials.Credentials) in client
# https://google-auth.readthedocs.io/en/stable/user-guide.html#service-account-private-key-files
# from google.oauth2 import service_account
# credentials = service_account.Credentials.from_service_account_file('/path/to/key.json')

# filename = '../artimage/multi-western/187124.jpg'
# response.full_text_annotation.pages[0].blocks[0]
# https://cloud.google.com/vision/docs/reference/rest/v1/images/annotate#TextAnnotation
# >>> response.text_annotations[0]
# locale: "sv"
# description: "HeLa\nMCF7\n15min\n120\n100\n85\nWB: Uba2\nHeLa\nU2OS\n100\n85\nWB: Uba2\nRPE1\n15min O\n100\n\344\270\200Uba2\n85\nWB: Uba2\n"
# bounding_poly {
#   vertices {
#     x: 62
#     y: 20
#   }
#   vertices {
#     x: 476
#     y: 20
#   }
#   vertices {
#     x: 476
#     y: 700
#   }
#   vertices {
#     x: 62
#     y: 700
#   }
# }

# >>> response.text_annotations[1]
# description: "HeLa"
# bounding_poly {
#   vertices {
#     x: 215
#     y: 22
#   }
#   vertices {
#     x: 274
#     y: 22
#   }
#   vertices {
#     x: 274
#     y: 41
#   }
#   vertices {
#     x: 215
#     y: 41
#   }
# }


# encoding and conversion of pieces of text identified by OCR
# process img through OCR service (start with Google Cloud Vision OCR)

# for each element identified find matches (threshold with Lehvenstein distance) along the text

# transform position (or bounding box) of element as position on a G x G grid
# position coded as index number of grid element, numbered 1 to G x G from top left to bottom right
# i =  x_grid * G + y_grid with x_grid = floor((G * (x / H))), y_grid = floor((G * (y / H)))
# fill position code along all matching characters
#    0  1  2  3  4
#   ---------------
# 0| 0  1  2  3  4 |
# 1| 5  6  7  8  9 |
# 2| 10 11 12 13 14|
# 3| 15 16 17 18 19|
# 4| 20 21 22 23 24|
#   ---------------
#    0  1  2  3  4
#   ---------------
# 0| .  .  .  .  . |
# 1| .  cat.  .  . |
# 2| .  .  .  .  . |
# 3| .  .  bat.  . |
# 4| .  .  .  .  . |
#   ---------------

# convert grid position using 1-hot encoding but fill with similarty score (Lehvenstein distance) to the text segment
# add horiz and vert 1-hot codes for orientation of the text
#
# with G =5
# grid_0  .0 .0 .0 .0 .0 .0 .0 .0 .0 .0 .0
# grid_1  .0 .0 .0 .0 .0 .0 .0 .0 .0 .0 .0
# grid_2  .0 .0 .0 .0 .0 .0 .0 .0 .0 .0 .0
# grid_3  .0 .0 .0 .0 .0 .0 .0 .0 .0 .0 .0
# ....
# grid_6  .0 .0 .0 .0 .9 .9 .9 .0 .0 .0 .0
# ....
# grid_17 .0 .0 .0 .0 .8 .8 .8 .0 .0 .0 .0
# ....
# grid_25 .0 .0 .0 .0 .0 .0 .0 .0 .0 .0 .0
# horiz   .0 .0 .0 .0 .1 .1 .1 .0 .0 .0 .0
# vert    .0 .0 .0 .0 .0 .0 .0 .0 .0 .0 .0
# text    t  h  e  _  c  a  t  _  a  t  e

import io
import os
import argparse
import json
import torch
from xml.etree.ElementTree  import XML, parse, tostring
from nltk.metrics.distance import edit_distance
import cv2 as cv
from google.cloud import vision
from google.cloud.vision import types
from google.oauth2 import service_account
from math import floor, ceil
from ..common.utils import cd
from ..common.progress import progress
from .. import config

ALLOWED_FILE_EXTENSIONS =['.jpg', '.jpeg']

class OCR():
    """
    [WIP] Will need to be included in encoder module?
    Extracts text elements from images and encode as features for relevant segments of the respective text examples.

    Args:
        path: path to the directory where xml documents and images can be found
        G: size of the grid, grid is G x G
        T: T similarity threshold (edit distance or Levenshtein distance / word length) when searching for best match between OCRed element and text
        accound_key: path to the account_key json file created in https://console.cloud.google.com/apis/credentials
    """

    def __init__(self, path, G=config.img_grid_size, T=0.1, account_key='smarttag-2-5b02d5e85409.json'):
        self.path = path # path to images
        # path to annotations
        self.client = vision.ImageAnnotatorClient(credentials = service_account.Credentials.from_service_account_file(account_key))

    def load_image(self, img_filename):
        try:
            with io.open(img_filename, 'rb') as image_file:
                img = image_file.read()
            img_cv = cv.imread(img_filename) # hack, would be better not to read file twice... dunno how to get image size otherwise!!
            shape = img_cv.shape
            h = shape[0]
            w = shape[1]
        except Exception as e:
            print("{} not loaded".format(img_filename))
            print(e)
            img = None
            h = 0
            w = 0
        return img, h, w

    def get_ocr(self, img):
        if img is not None:
            image = types.Image(content=img)
            # Performs text detection on the image file; see https://googleapis.github.io/google-cloud-python/latest/vision/gapic/v1/api.html#google.cloud.vision_v1.ImageAnnotatorClient.text_detection
            response = self.client.text_detection(image)
            annotations = OCRAnnotations.from_google(response.text_annotations)
        else:
            annotations = None
        return annotations

    def save_annotations(self, annotations, filename):
        with open(filename, 'w') as f:
           json.dump(annotations.dict, f) # json.dump(annotations, f, cls=myJSONEncoder)

    def run(self):
        with cd(self.path):
            filenames = [f for f in os.listdir() if os.path.splitext(f)[-1] in ALLOWED_FILE_EXTENSIONS]
            for filename in filenames:
                basename = os.path.splitext(filename)[0]
                ocr_filename = basename +'.json'
                empty = True
                if os.path.exists(ocr_filename):
                    with open(ocr_filename, 'r') as f:
                        content = f.read()
                    empty = (content == "")
                if empty:
                    img, h, w = self.load_image(filename)
                    annotations = self.get_ocr(img)
                    annotations.image_width = w
                    annotations.image_height = h
                    print('\nTextual elements detected on image {}:'.format(filename))
                    print(", ".join(['"'+a.text+'"' for a in annotations]))
                    self.save_annotations(annotations, ocr_filename)
                else:
                    # add a check of whether file empty, can happen rarely
                    print("{} has already been OCR-ed".format(filename))

class OCRAnnotations(object):

    def __init__(self, annot_dict):
        self._annotations = []
        self._image_height = None
        self._image_width = None
        if annot_dict:
            self._annotations = [Annotation(a) for a in annot_dict['annotations']]
            self._image_height = annot_dict['image_height']
            self._image_width = annot_dict['image_width']

    @classmethod
    def from_google(cls, google_annot):
        annotation_list = [Annotation.from_google(a) for a in google_annot[1:]]
        annot_dict = [a.dict for a in annotation_list]
        d = {
            'image_width': None,
            'image_height': None,
            'annotations': annot_dict
        }
        return cls(d)

    def __iter__(self):
        return self._annotations.__iter__()

    def __next__(self):
        return next(self._annotations)

    def __len__(self):
        return len(self._annotations)

    def __getitem__(self, i):
        return self._annotations[i]

    @property
    def image_height(self):
        return self._image_height
    @property
    def image_width(self):
        return self._image_width
    @image_height.setter
    def image_height(self, x):
        self._image_height = x
    @image_width.setter
    def image_width(self, x):
        self._image_width = x

    @property
    def dict(self):
        d = {
            'image_width': self.image_width,
            'image_height': self.image_height,
            'annotations': [a.dict for a in self._annotations]
        }
        return d


class Annotation(object):

    def __init__(self, annot):
        self._annot = annot
        self._text = annot['text']
        self._coordinates =  annot['coordinates']
        self._orientation =  annot['orientation']

    @classmethod
    def from_google(cls, google_annot):
        annot = {
             'text': cls._set_text(google_annot),
             'coordinates': cls._set_coordinates(google_annot),
             'orientation': cls._set_orientation(google_annot)
        }
        return cls(annot)

    @property
    def dict(self):
        return self._annot

    @property
    def text(self):
        return self._text

    @property
    def coordinates(self):
        return self._coordinates

    @property
    def orientation(self):
        return self._orientation

    @staticmethod
    def _set_text(google_annot):
        return google_annot.description

    @staticmethod
    def _set_coordinates(google_annot):
        return (google_annot.bounding_poly.vertices[0].x, google_annot.bounding_poly.vertices[0].y)

    @staticmethod
    def _set_orientation(google_annot):
        """
        when the text is horizontal it might look like:
        0----1
        |    |
        3----2
        when it's rotated 180 degrees around the top-left corner it becomes:
        2----3
        |    |
        1----0
        and the vertice order will still be (0, 1, 2, 3).
        """
        epsilon = 5
        x0 = google_annot.bounding_poly.vertices[0].x
        y0 = google_annot.bounding_poly.vertices[0].y
        x1 = google_annot.bounding_poly.vertices[1].x
        y1 = google_annot.bounding_poly.vertices[1].y
        if abs(y1 - y0) < epsilon:
            orientation = 'horizontal'
        elif abs(x1 - x0) < epsilon:
            orientation = 'vertical'
        else:
            orientation  = 'unknown'
        print(google_annot.description, x0, y0, x1, y1, orientation) # STOD 362 307 442 321 unknown
        return orientation


class OCREncoder(object):

    def __init__(self, path, G=config.img_grid_size, T=config.ocr_max_edit_dist, L=config.ocr_min_overlap):
        self.path = path
        self.G = G
        self.edit_threshold = T
        self.min_overlap = L


    def grid_index(self, h, w, annot):
        """
        Transforms coordinates into an indexed grid position.
        Position indexed from top left to right bottom

        Args:
            h, w : height and width of the image
            annot: ocr annotation
        """

        x, y = annot.coordinates
        row = floor(self.G * (y / h))
        column = floor(self.G * (x / w))
        grid_pos = (row * self.G) + column
        # compress this into binary code
        return grid_pos

    def best_matches(self, text, annot):
        query = annot.text
        l = len(query)
        L = len(text)
        matches = []
        for i in range(L-l):
            dist = edit_distance(text[i:i+l].lower(), query.lower()) # Levenshtein distance between query and text at this position
            dist /= l # distance per word length
            if dist <= self.edit_threshold and l >= self.min_overlap:
                matches.append({
                    'match': text[i:i+l],
                    'query': query,
                    'length': l,
                    'score': 1-dist,
                    'pos': i
                })
        return matches

    def add_context_(self, context_tensor, pos_on_grid, orientation, pos_in_text, length, score):
        # 1-hot encoding of position on the grid
        context_tensor[pos_on_grid, pos_in_text:pos_in_text+length] = score
        if orientation == 'horizontal':
            context_tensor[self.G ** 2, pos_in_text:pos_in_text+length] = 1
        elif orientation == 'vertical':
            context_tensor[self.G ** 2 + 1, pos_in_text:pos_in_text+length] = 1
        elif orientation == 'unkown':
            context_tensor[self.G ** 2,     pos_in_text:pos_in_text+length] = 0.5
            context_tensor[self.G ** 2 + 1, pos_in_text:pos_in_text+length] = 0.5

    def load_annotations(self, filename):
        j = {}
        with cd(self.path):
            if os.path.exists(filename):
                with open(filename, 'r') as f:
                    try:
                        j = json.load(f)
                    except Exception as e:
                        j = {}
                        print("problem with json file", filename, e)
        return OCRAnnotations(j)

    def encode(self, text, graphic_filename):
        basename = os.path.splitext(graphic_filename)[0]
        ocr_filename = basename + ".json"
        annotations = self.load_annotations(ocr_filename)
        h = annotations.image_height
        w = annotations.image_width
        context_tensor = torch.zeros(self.G ** 2 + 2, len(text)) # G^2 1-hot encoded features for position on the grid + 2 features for horizontal vs vertical orientation
        for annot in annotations: # first annoation is the full list of entities detected
            pos_on_grid = self.grid_index(h, w, annot)
            matches = self.best_matches(text, annot)
            orientation = annot.orientation
            for m in matches:
                self.add_context_(context_tensor, pos_on_grid, orientation, m['pos'], m['length'], m['score'])
        return context_tensor

def main():
    parser = config.create_argument_parser_with_defaults(description='Modules to perform OCR and encode OCR-based context.')

    args = parser.parse_args()
    print("running ocr from",  os.getcwd(), config.image_dir)
    ocr = OCR(config.image_dir, config.img_grid_size)
    ocr.run()

if __name__ == '__main__':
    main()
