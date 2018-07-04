# -*- coding: utf-8 -*-
#T. Lemberger, 2018


import unittest
import torch
from test.smtagunittest import SmtagTestCase
from smtag.mapper import Catalogue, Concept, Category, Entity, Role, Boundary

class MapperTest(SmtagTestCase):


    def test_add_1(self):
        gene_protein = Catalogue.GENE + Catalogue.PROTEIN
        self.assertEqual("gene_protein", gene_protein.type)
    
    def test_add_2(self):
        gene_intervention = Catalogue.GENE + Catalogue.INTERVENTION
        self.assertEqual("gene", gene_intervention.type)
        self.assertEqual("intervention", gene_intervention.role)
        self.assertEqual("entity", gene_intervention.category)

    def test_add_3(self):
        geneprod_reporter = Catalogue.GENEPROD + Catalogue.REPORTER
        #geneprod_reporter = Catalogue.REPORTER + Catalogue.GENEPROD
        self.assertEqual("geneprod", geneprod_reporter.type)
        self.assertEqual("reporter", geneprod_reporter.role)
        self.assertEqual("entity", geneprod_reporter.category)

    def test_find_index(self):
        gene_intervention = Catalogue.GENE + Catalogue.INTERVENTION
        concept_list = [Catalogue.PROTEIN, gene_intervention, Catalogue.TISSUE]
        index_protein = Catalogue.PROTEIN.my_index(concept_list)
        index_gene_intervention = gene_intervention.my_index(concept_list)
        index_gene = Catalogue.GENE.my_index(concept_list)
        index_tissue = Catalogue.TISSUE.my_index(concept_list)
        not_found = Catalogue.CELL.my_index(concept_list)
        self.assertEqual(0, index_protein)
        self.assertEqual(1, index_gene_intervention)
        self.assertEqual(1, index_gene)
        self.assertEqual(2, index_tissue)
        self.assertEqual(None, not_found)


if __name__ == '__main__':
    unittest.main()

