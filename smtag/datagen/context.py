# install gcloud for mac https://cloud.google.com/sdk/docs/quickstart-macos
# of linux https://cloud.google.com/sdk/docs/quickstart-linux
# see also https://googleapis.github.io/google-cloud-python/latest/vision/index.html
# https://googleapis.github.io/google-cloud-python/latest/vision/
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

# filename = '../artimage/multi-western/187124.jpg'
# response.full_text_annotation.pages[0].blocks[0]
# property {
#   detected_languages {
#     language_code: "sv"
#   }
# }
# bounding_box {
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
# paragraphs {
#   property {
#     detected_languages {
#       language_code: "sv"
#     }
#   }
#   bounding_box {
#     vertices {
#       x: 215
#       y: 22
#     }
#     vertices {
#       x: 274
#       y: 22
#     }
#     vertices {
#       x: 274
#       y: 41
#     }
#     vertices {
#       x: 215
#       y: 41
#     }
#   }
#   words {
#     property {
#       detected_languages {
#         language_code: "sv"
#       }
#     }
#     bounding_box {
#       vertices {
#         x: 215
#         y: 22
#       }
#       vertices {
#         x: 274
#         y: 22
#       }
#       vertices {
#         x: 274
#         y: 41
#       }
#       vertices {
#         x: 215
#         y: 41
#       }
#     }
#     symbols {
#       property {
#         detected_languages {
#           language_code: "sv"
#         }
#       }
#       bounding_box {
#         vertices {
#           x: 215
#           y: 22
#         }
#         vertices {
#           x: 229
#           y: 22
#         }
#         vertices {
#           x: 229
#           y: 41
#         }
#         vertices {
#           x: 215
#           y: 41
#         }
#       }
#       text: "H"
#     }
#     symbols {
#       property {
#         detected_languages {
#           language_code: "sv"
#         }
#       }
#       bounding_box {
#         vertices {
#           x: 232
#           y: 27
#         }
#         vertices {
#           x: 245
#           y: 27
#         }
#         vertices {
#           x: 245
#           y: 41
#         }
#         vertices {
#           x: 232
#           y: 41
#         }
#       }
#       text: "e"
#     }
#     symbols {
#       property {
#         detected_languages {
#           language_code: "sv"
#         }
#       }
#       bounding_box {
#         vertices {
#           x: 248
#           y: 22
#         }
#         vertices {
#           x: 260
#           y: 22
#         }
#         vertices {
#           x: 260
#           y: 41
#         }
#         vertices {
#           x: 248
#           y: 41
#         }
#       }
#       text: "L"
#     }
#     symbols {
#       property {
#         detected_languages {
#           language_code: "sv"
#         }
#         detected_break {
#           type: EOL_SURE_SPACE
#         }
#       }
#       bounding_box {
#         vertices {
#           x: 262
#           y: 27
#         }
#         vertices {
#           x: 274
#           y: 27
#         }
#         vertices {
#           x: 274
#           y: 41
#         }
#         vertices {
#           x: 262
#           y: 41
#         }
#       }
#       text: "a"
#     }
#   }
# }
# block_type: TEXT

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
import torch
from nltk.metrics.distance import edit_distance
import cv2 as cv
from google.cloud import vision
from google.cloud.vision import types
from math import floor
from common.utils import cd
from .. import config

ALLOWED_FILE_EXTENSIONS =['txt', 'jpg']

class OCR():
    """
    Extracts text elements from images and encode as features for relevant segments of the respective text examples.
    """

    G = 5 # grid is G x G
    T = 0.8 # similarity threshold when searching for best match between OCRed element and text

    def __init__(self, path):
        self.client = vision.ImageAnnotatorClient()
        self.examples = self.load_examples()
        self.N = len(self.examples)
        self.T = torch.Tensor(self.N, self.G, len(text))

    def load_from_dir(self, path):
        xml_documents = self.import_files(path)
        examples = self.load_examples(xml_documents)
        return examples
    
    def load_examples(self, examples):
        """
        Encodes examples provided as XML Elements.
        """

        encoded_examples = []

        for id in examples:
            
            if text:
                example = examples[id]
                img_filename = example.attrib['img']
                #encoded_features, _, _ = XMLEncoder.encode(example)
                example = {
                    'provenance': id,
                    'text': text,
                    'img'; img_filename
                }
                encoded_examples.append(example)
            else:
                print("\nskipping an example in document with id=", id)
                print(tostring(examples[id]))
        return encoded_examples


    def import_files(self, XPath_to_examples='.//sd-panel'):
        """
        Import xml documents from dir. In each document, extracts examples using XPath.
        """

        with cd(config.data_dir):
            # path = os.path.join(self.compendium, subset)
            print("\nloading from:", path)
            os.
            filenames = [f for f in os.listdir(path) if os.path.splitext(f)[-1] == 'xml']
            examples = {}
            for i, filename in enumerate(filenames):
                try:
                    with open(os.path.join(path, filename)) as f: 
                        xml = parse(f)
                    for j, e in enumerate(xml.findall(XPath_to_examples)):
                        id = filename + "-" + str(j) # unique id provided filename is unique (hence limiting to single allowed file extension)
                        examples[id] = e
                except Exception as e:
                    print("problem parsing", os.path.join(path, filename))
                    print(e)
                progress(i, len(filenames), "loaded {}".format(filename))
        return examples

    def get_example(self, i):
        text = self.examples[i]['text']
        img_filename = self.examples[i]['img']
        with io.open(img_filename, 'rb') as image_file:
            img = image_file.read()
            img_cv = cv.read(image_file)
            shape = img_cv.shape()
        return text, img, shape
    
    def get_ocr(self, img):

        # Loads the image into memory

        image = types.Image(content=img)

        #extract image dimensions HERE

        # Performs text detection on the image file
        # https://googleapis.github.io/google-cloud-python/latest/vision/gapic/v1/api.html#google.cloud.vision_v1.ImageAnnotatorClient.text_detection
        # text_detection(image, max_results=None, retry=None, timeout=None, **kwargs)#
        response = self.client.text_detection(image)
        annotations = response.text_annotations
        print('Labels:')
        for a in annotations:
            print(a.description)
        return annotations

    def get_coordinates(self, annotation):
        return annotations.bounding_poly.vertices[0]
    
    def get_orientation(self, annotation):
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
        x0 = annotation.bounding_poly.vertices[0].x0
        x1 = annotation.bounding_poly.vertices[1].x
        if x1 > x0:
            return 'horizontal'
        elif x1 < x0:
            return 'vertical'
        else:
            return 'unknown'

    def get_text(self, element):
        return annotation.description

    def grid_index(self, h, w, element):
        """transforms coordinates into indexed grid position"""
        x, y = self.get_coordinates(element)
        grid_pos = (floor(self.G * (y / h)) * self.G) + floor(self.G * (x / w))
        return grid_pos

    def best_matches(self, text, element):
        query = self.get_text(element)
        l = len(query)
        L = len(text)
        matches = []
        for i in range(L-l):
            dist = edit_distance(text[i:i+l], query) # Levenshtein distance between query and text at this position
            if dist <= T:
                matches.append({
                    'match': text[i:i+l],
                    'length': l,
                    'dist': dist,
                    'pos': i
                })
        return matches


    def add_context(self, pos_on_grid, pos_in_text, length, score):
        self.T[pos_on_grid, pos_in_text:pos_in_text+length] = score

    def run(self):
        for i, filename in enumerate(self.filenames):
            text, img, shape = self.get_example(i)
            h = shape[0]
            w = shape[1]
            annotations = self.get_ocr(img)
            for a in annotations:
                pos_on_grid = self.grid_index(h, w, a) # 
                matches = self.best_matches(text, a)
                for pos_in_text in matches:
                    self.add_context(pos_on_grid, pos_in_text['pos'], pos_in_text['length'], pos_in_text['dist'])
