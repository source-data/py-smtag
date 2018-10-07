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

ALLOWED_FILE_EXTENSIONS =['xml', 'jpg']

class OCR():
    """
    [WIP] Will need to be included in encoder module?
    Extracts text elements from images and encode as features for relevant segments of the respective text examples.

    Args:
        path: path to the directory where xml documents and images are found
        G: size of the grid, grid is G x G
        T: T similarity threshold (edit distance or Levenshtein distance / word length) when searching for best match between OCRed element and text
        accound_key: path to the account_key json file created in https://console.cloud.google.com/apis/credentials
    """

    def __init__(self, path, G=5, T=0.1, account_key='/Users/lemberger/Documents/code/cloud/smarttag-2-5b02d5e85409.json'):
        self.path = path
        self.G = G
        self.T = T
        self.client = vision.ImageAnnotatorClient(credentials = service_account.Credentials.from_service_account_file(account_key))
        self.examples = self.load_from_dir()
        self.N = len(self.examples)
        self.context_codes = []
        #self.context_tensor = torch.Tensor(self.N, self.G, len(text))

    def load_from_dir(self):
        xml_documents = self.import_xml_files()
        examples = self.load_examples(xml_documents)
        return examples
    
    def load_examples(self, examples):
        """
        Encodes examples provided as XML Elements.
        """

        encoded_examples = []

        for id in examples:
            example = examples[id]
            text = ''.join([s for s in example.itertext()])
            if text:
                img_filename = example.attrib['img']
                #encoded_features, _, _ = XMLEncoder.encode(example)
                #encoded_context = OCR.encode(example) #or somethign like that in the future 
                example = {
                    'provenance': id,
                    'text': text,
                    'img': img_filename
                }
                encoded_examples.append(example)
            else:
                print("\nskipping an example in document with id=", id)
                print(tostring(examples[id]))
        return encoded_examples


    def import_xml_files(self, XPath_to_examples='.//sd-panel'):
        """
        Import xml documents from path. In each document, extracts examples using XPath. 
        """

        with cd(self.path):
            # path = os.path.join(self.compendium, subset)
            print("\nloading from:", self.path)
            filenames = os.listdir()
            filenames = [f for f in filenames if f.split('.')[-1] == 'xml']
            print("filenames", filenames)
            examples = {}
            for i, filename in enumerate(filenames):
                try:
                    with open(filename) as f: 
                        xml = parse(f)
                    for j, e in enumerate(xml.findall(XPath_to_examples)):
                        id = filename + "-" + str(j) # unique id provided filename is unique (hence limiting to single allowed file extension)
                        examples[id] = e
                except Exception as e:
                    print("problem parsing", filename)
                    print(e)
                progress(i, len(filenames), "loaded {}".format(filename))
        return examples

    def get_example(self, i):
        text = self.examples[i]['text']
        img_filename = self.examples[i]['img']
        with cd(self.path):
            with io.open(img_filename, 'rb') as image_file:
                img = image_file.read()
            img_cv = cv.imread(img_filename) # hack, would be better not to read file twice... dunno how to get image size otherwise!!
        shape = img_cv.shape
        h = shape[0]
        w = shape[1]
        return text, img, h, w

    def get_ocr(self, img):
        image = types.Image(content=img)
        # Performs text detection on the image file; see https://googleapis.github.io/google-cloud-python/latest/vision/gapic/v1/api.html#google.cloud.vision_v1.ImageAnnotatorClient.text_detection
        response = self.client.text_detection(image)
        annotations = response.text_annotations
        print('\nTextual elements detected on image:')
        for a in annotations:
            print(a.description)
        print("###########")
        return annotations

    def get_coordinates(self, annot):
        return annot.bounding_poly.vertices[0].x, annot.bounding_poly.vertices[0].x
    
    def get_orientation(self, annot):
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
        x0 = annot.bounding_poly.vertices[0].x
        x1 = annot.bounding_poly.vertices[1].x
        if x1 > x0:
            return 'horizontal'
        elif x1 < x0:
            return 'vertical'
        else:
            return 'unknown'

    def get_text(self, annot):
        return annot.description

    def grid_index(self, h, w, annot):
        """
        Transforms coordinates into an indexed grid position.
        Position indexed from top left to right bottom
        
        Args:
            h, w : height and width of the image
            annot: ocr annotation
        """
        x, y = self.get_coordinates(annot)
        grid_pos = (floor(self.G * (y / h)) * self.G) + floor(self.G * (x / w))
        return grid_pos

    def best_matches(self, text, annot):
        query = self.get_text(annot)
        l = len(query)
        L = len(text)
        matches = []
        for i in range(L-l):
            dist = edit_distance(text[i:i+l], query) # Levenshtein distance between query and text at this position
            dist /= l # distance per word length
            if dist <= self.T:
                matches.append({
                    'match': text[i:i+l],
                    'length': l,
                    'score': 1-dist,
                    'pos': i
                })
        return matches

    def add_context_(self, context_tensor, pos_on_grid, pos_in_text, length, score):
        context_tensor[pos_on_grid, pos_in_text:pos_in_text+length] = score

    def encode_ocr(self, text, h, w, annotations):
        context_tensor = torch.zeros(self.G ** 2, len(text))
        for annot in annotations:
            pos_on_grid = self.grid_index(h, w, annot)
            matches = self.best_matches(text, annot)
            for m in matches:
                print(m)
                self.add_context_(context_tensor, pos_on_grid, m['pos'], m['length'], m['score'])
        return context_tensor
    
    def run(self):
        for i in range(self.N): # for text, img, shape in self.examples # define class Examples and class Example
            text, img, h, w = self.get_example(i)
            annotations = self.get_ocr(img)
            encoded_ocr_context = self.encode_ocr(text, h, w, annotations)
            self.context_codes.append(encoded_ocr_context)


def main():
    test_path = os.path.join(config.data_dir, 'test_img', 'train')
    ocr = OCR(test_path)
    ocr.run()
    print(ocr.examples)
    for i, t in enumerate(ocr.context_codes):
        print(ocr.examples[i]['text'])
        for j in range(t.size(0)):
            print("".join([['-','+'][ceil(s)] for s in t[j]]))


if __name__ == '__main__':
    main()