import unittest
import torch
from smtag import loader, config

class ParsingDatasetTest(unittest.TestCase):
    def setUp(self):
        self.dataset = loader.Dataset('test_train')
        self.number_of_examples = 780
        self.length_of_examples = 170
        self.number_of_features = 24

    def test_calculates_number_of_examples_from_numpy_tensor(self):
        self.assertEqual(self.dataset.N, self.number_of_examples)

    def test_calculates_size_of_examples_from_numpy_tensor(self):
        self.assertEqual(self.dataset.L, self.length_of_examples)

    def test_calculates_number_of_features_from_numpy_tensor(self):
        self.assertEqual(self.dataset.nf, self.number_of_features)

    def test_converts_numpy_tensor_to_torch(self):
        self.assertEqual(type(self.dataset.features), torch.DoubleTensor)

    def test_reshapes_the_numpy_tensor_to_4_dimensions(self):
        new_shape = (self.number_of_examples,self.number_of_features,1,self.length_of_examples)
        self.assertEqual(self.dataset.features.size(), new_shape)

    def test_reads_the_original_text_of_datasets(self):
        self.assertEqual(len(self.dataset.text), self.number_of_examples)

    @unittest.skip("needs refactor to facilitate testing")
    def test_checks_the_length_of_every_example_while_reading_them(self):
        pass

    def test_reads_the_provenance_of_each_example(self):
        self.assertEqual(len(self.dataset.provenance), self.number_of_examples)


class LoaderInitTest_Defaults(unittest.TestCase):
    def setUp(self):
        self.selected_features = ['potein', 'gene']
        self.loader = loader.Loader(self.selected_features)

    def test_has_sensible_defaults(self):
        self.assertEqual(self.loader.collapsed_features, [])
        self.assertEqual(self.loader.overlap_features, {})
        self.assertEqual(self.loader.features2input, [])
        self.assertEqual(self.loader.noise, 0)
        self.assertEqual(self.loader.fraction, 1)
        self.assertEqual(self.loader.validation_fraction, 0.2)
        self.assertEqual(self.loader.nf_input, config.NBITS)
        self.assertEqual(self.loader.nf_collapsed_feature, 0)
        self.assertEqual(self.loader.nf_overlap_feature, 0)

class LoaderInitTest_nf_input(unittest.TestCase):
    def setUp(self):
        self.selected_features = ['potein', 'gene']
        self.features2input = ['zzz']
        self.loader = loader.Loader(self.selected_features, features2input=self.features2input)

    def test_nf_input_depends_of_config_and_features2input(self):
        self.assertEqual(self.loader.nf_input, config.NBITS + len(self.features2input))

class LoaderInitTest_nf_output(unittest.TestCase):
    def test_deafults_to_the_number_of_selected_features(self):
        selected_features = ['potein', 'gene']
        ldr = loader.Loader(selected_features)
        self.assertEqual(ldr.nf_output, len(selected_features))

    def test_gets_increased_by_1_if_there_are_any_collapsed_features(self):
        selected_features = ['potein', 'gene']
        collapsed_features = ['xxx','zzz']
        ldr = loader.Loader(selected_features, collapsed_features=collapsed_features)
        self.assertEqual(ldr.nf_output, len(selected_features) + 1)

    def test_gets_increased_by_1_if_there_are_any_overlap_features(self):
        selected_features = ['potein', 'gene']
        overlap_features = ['xxx','zzz']
        ldr = loader.Loader(selected_features, overlap_features=overlap_features)
        self.assertEqual(ldr.nf_output, len(selected_features) + 1)

    def test_equals_nf_input_otherwise(self):
        ldr = loader.Loader([])
        self.assertEqual(ldr.nf_output, ldr.nf_input)


