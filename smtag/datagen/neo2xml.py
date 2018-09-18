# -*- coding: utf-8 -*-
#T. Lemberger, 2018

import re
import os
import argparse
from getpass import getpass
from xml.etree.ElementTree import fromstring, Element, SubElement, tostring
from neo4jrestclient.client import GraphDatabase, Node
from random import shuffle
from math import floor
import difflib
from ..common.utils import cd
from ..common.config import MARKING_CHAR, MARKING_CHAR_ORD  
from .. import config

SD_PANEL_OPEN, SD_PANEL_CLOSE = "<sd-panel>", "</sd-panel>"

class NeoImport():
    
    def __init__(self, options):
        self.options = options
        self.articles = {}


    @staticmethod
    def caption_text2xml(panel_caption, tags, tags2anonym, safe_mode = True, exclusive_mode = False, keep_roles_only_for_selected_tags = False):
        def anonymize_sdtags(panel_xml, tags_neo):
            # mini_index = [] #keeps an index of terms that are the same; will be anonymized by respective 'marker' characters
            for t in tags_neo:
                tag_xml = panel_xml.find('.//sd-tag[@id="sdTag{}"]'.format(t['data']['id']))
                if tag_xml is not None:
                    if tag_xml.text:
                        # tag_text_lo = tag_xml.text.lower()
                        # if tag_text_lo not in mini_index: mini_index.append(tag_text_lo)
                        # mark = unichr(MARKING_CHAR_ORD + mini_index.index(tag_text_lo))
                        # tag_xml.text = mark * len(tag_xml.text)
                        tag_xml.text = MARKING_CHAR * len(tag_xml.text)
            return panel_xml
        
        tag_errors = []
        panel_caption = panel_caption#.encode('utf-8')
        if safe_mode:
            #need protection agains missing spaces

            #protection against carriage return
            if re.search('[\r\n]', panel_caption):
                print("WARNING: removing return characters")
                panel_caption = re.sub('[\r\n]','', panel_caption)

            #protection against <br> instead of <br/>
            panel_caption = re.sub(r'<br>', r'<br/>', panel_caption)

            #protection against badly formed link elements
            panel_caption = re.sub(r'<link href="(.*)">', r'<link href="\1"/>', panel_caption)
            panel_caption = re.sub(r'<link href="(.*)"/>(\n|.)*</link>', r'<link href="\1">\2</link>', panel_caption)
            #protection agains missing <sd-panel> tags
            if re.search(r'^{}(\n|.)*{}$'.format(SD_PANEL_OPEN, SD_PANEL_CLOSE), panel_caption) is None:
                print("WARNING: correcting missing <sd-panel> </sd-panel> tags!")
                print(panel_caption)
                panel_caption = SD_PANEL_OPEN + panel_caption + SD_PANEL_CLOSE


        #We may loose a space that separates panels in the actual figure legend...
        panel_caption = re.sub('</sd-panel>$', ' </sd-panel>', panel_caption)
        #and then remove possible runs of spaces
        panel_caption = re.sub(r' +',r' ', panel_caption)

        panel_xml = fromstring(panel_caption)

        tags_xml = panel_xml.findall('.//sd-tag')
        tags_neo_id = [u"sdTag{}".format(t['data']['id']) for t in tags]
        tags_not_found = set(tags_neo_id) - set([t.attrib['id'] for t in tags_xml])
        if tags_not_found:
            print("WARNING, tag(s) not found: ", tags_not_found)
            print(panel_caption)
            tag_errors.append(tags_not_found)

        #keep attributes only for the selected tags and clear the rest
        if exclusive_mode:
            for t_xml in tags_xml:
                if 'id' in t_xml.attrib:
                    if not t_xml.attrib['id'] in tags_neo_id:
                        t_xml.attrib.clear()
                else:
                    print("WARNING, tag", tostring(t_xml), "has no id" )

        if keep_roles_only_for_selected_tags:
            for t_xml in tags_xml:
                if 'id' in t_xml.attrib and 'role' in  t_xml.attrib:
                    if not t_xml.attrib['id'] in tags_neo_id:
                        t_xml.attrib.pop('role')

        #anonymize a subset of the tags
        if tags2anonym:
            anonymize_sdtags(panel_xml, tags2anonym)

        return panel_xml, tag_errors

    def neo2xml(self, source):

        where_clause = self.options['where_clause']
        entity_type_clause = self.options['entity_type_clause']
        entity_role_clause = self.options['entity_role_clause']
        tags2anonmymize_clause = self.options['tags2anonmymize_clause']
        donotanonymize_clause = self.options['donotanonymize_clause']
        limit_clause = self.options['limit_clause']
        safe_mode = self.options['safe_mode']
        exclusive_mode = self.options['exclusive_mode']
        keep_roles_only_for_selected_tags = self.options['keep_roles_only_for_selected_tags']

        DB = GraphDatabase(source['db'],source['username'], source['password'])

        caption_errors = []
        tag_level_errors = []
        paper_errors = []

        q_articles = '''
        MATCH (a:Article)
        {} //WHERE clause
        RETURN id(a), a.doi
        {} //LIMIT clause
        '''.format(where_clause, limit_clause)

        results_articles = DB.query(q_articles)

        counter = 0
        for a in results_articles:
            a_id = a[0]
            doi = a[1]
            if doi == '': doi = a_id
            if doi in self.articles:
                print('WARNING! {} ALREADY EXISTS'.format(doi))
                paper_errors.append({a_id, doi})
            else:
                q_figures = '''
                    MATCH (a:Article )-->(f:Figure)
                    WHERE id(a) = {}
                    RETURN id(f), f.fig_label //, f.caption
                    ORDER BY f.fig_label ASC
                    '''.format(a_id)
                results_figures = DB.query(q_figures)

                self.articles[doi] = Element('Article')
                self.articles[doi].attrib['doi'] = doi

                for f in results_figures:
                    f_id = f[0]
                    fig_label = f[1]

                    q_panel = '''
                    MATCH (f:Figure)-->(p:Panel)-->(t:Tag)
                    WHERE id(f) = {} AND t.in_caption = true
                    {} //AND t.type = some_entity_type OR some other type
                    {} //AND t.role = some role OR some role
                    WITH p.formatted_caption AS formatted_caption, p.label AS label, p.panel_id AS panel_id, COLLECT(DISTINCT t) AS tags
                    RETURN formatted_caption, label, panel_id, tags , [t in tags WHERE (t.type in [{}] AND NOT t.role in[{}])] AS tags2anonym // (t.type in ["gene","protein"] AND NOT t.role in ["reporter"])

                    '''.format(f_id, entity_type_clause, entity_role_clause, tags2anonmymize_clause, donotanonymize_clause)
                    results_panels = DB.query(q_panel)
                    print((u"{} panels found for figure {} ({}) in paper {}".format(len(results_panels), fig_label, f_id, doi)).encode('utf-8'))

                    if results_panels:
                        figure_xml_element = Element('figure-caption')
                        #panels not in the proper order, need resorting via label
                        results_labeled = {p[1]:{'panel_caption':p[0], 'panel_id':p[2], 'fig_label':fig_label, 'tags':p[3], 'tags2anonym':p[4]} for p in results_panels}
                        sorted_panel_labels = list(results_labeled.keys())
                        sorted_panel_labels.sort()

                        for p in sorted_panel_labels:
                            panel_caption = results_labeled[p]['panel_caption']
                            tags = results_labeled[p]['tags']
                            tags2anonym = results_labeled[p]['tags2anonym']
                            try:
                                panel_xml_element, tag_errors = self.caption_text2xml(panel_caption, tags, tags2anonym, safe_mode, exclusive_mode, keep_roles_only_for_selected_tags)
                                figure_xml_element.append(panel_xml_element)
                                if tag_errors:
                                    panel_id = results_labeled[p]['panel_id']
                                    fig_label = results_labeled[p]['fig_label']
                                    tag_level_errors.append([doi, fig_label, p, panel_id, panel_caption, tag_errors])
                                counter += 1
                            except Exception as e:
                                panel_id = results_labeled[p]['panel_id']
                                fig_label = results_labeled[p]['fig_label']
                                print((u"problem parsing fig {} panel {} (panel_id:{}) in article {}".format(fig_label, p, panel_id, doi)).encode('utf-8'))
                                print(panel_caption.encode('utf-8'))
                                print(" ==> error: ", e, "\n")
                                caption_errors.append([doi, fig_label, p, panel_id, e, panel_caption])

                        self.articles[doi].append(figure_xml_element)

                print("number of figures in ", a_id, doi, len(self.articles[doi].getchildren()))
                print("counted {} panels.".format(counter))

        return {'paper_level':paper_errors, 'caption_level': caption_errors, 'tag_level': tag_level_errors} #figure_captions_text

    def split_dataset(self, testfrac):
        shuffled_doi = list(self.articles.keys())
        shuffle(shuffled_doi)
        N = len(shuffled_doi)
        end = floor(N * (1 - testfrac))
        trainset = []
        testset = []
        for i in range(0, end):
            doi = shuffled_doi[i]
            trainset.append(self.articles[doi])
        for i in range(end, N):
            doi = shuffled_doi[i]
            testset.append(self.articles[doi])
        return trainset, testset

    def save(self, split_dataset, data_dir, namebase):
        print("current dir:", os.curdir, os.listdir(os.curdir))
        with cd(config.data_dir):
            print("saving to: ", data_dir)
            if namebase in os.listdir():
                print("data {} already exists".format(namebase))
                raise(Exception())
            else:
                os.mkdir(namebase)
                with cd(namebase):
                    for subdir in split_dataset:
                        articles = split_dataset[subdir]
                        os.mkdir(subdir) # should we use os.chmod(os.mkdir(os.path.join(stock, subdir), 0o777)with iopen(os.path.join(self.path, str(id)+".jpg"), 'wb') as file:
                        for i, article in enumerate(articles):
                            filename = str(i)+'.xml'
                            print('writing to {}'.format(filename))
                            with open(os.path.join(subdir, filename), 'wb') as f:
                                f.write(tostring(article))

    def log_errors(self, errors):
        """
        Errors that are detected during feature extraction are kept and logged into a log file.
        """
        for e in errors:
            if errors[e]:
                print("####################################################")
                print(" Writing {} {} errors to errors_{}.log".format(len(errors[e]), e, e))
                print("####################################################" )
            #write log file anyway, even if zero errors, to remove old copy
            with open('errors_{}.log'.format(e), 'w') as f:
                for line in errors[e]:
                    ids, err = line
                    f.write(u"\nerror:\t{}\t{}\n".format('\t'.join(ids), err))
            f.close()

def main():
    parser = argparse.ArgumentParser(description='Top level module to manage training.', formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('-v', '--verbose', action='store_true', help='verbose mode.')
    parser.add_argument('-f', '--namebase', default='test_data', help='The name of the dataset')
    parser.add_argument('-A', '--tags2anonymize', default='', help='tag type to anonymise')
    parser.add_argument('-AA','--donotanonymize', default='', help='role of tags that should NOT be anonymized')
    parser.add_argument('-l', '--limit', default=10, type=int, help='limit number of papers scanned, mainly for testing')
    parser.add_argument('-Y', '--year_range', default='', help='select papers published in the start:end year range')
    parser.add_argument('-J', '--journals', default='', help='select set of journals, comma delimited')
    parser.add_argument('-D', '--doi', default='', help='restrict to a single doi')
    parser.add_argument('-y', '--type', default='', help='makes sure each example has entities of the specified type')
    parser.add_argument('-e', '--exclusive', action='store_true', help='only the tags selected by -y are kept, the others are removed')
    parser.add_argument('-s', '--selective', action='store_true', help='keep the roles only for the entities selected by the -y option')
    parser.add_argument('-r', '--role', default='', help='makes sure each example has entities with the specified role')
    parser.add_argument('-N', '--not_safe_mode', action='store_true', help='protects against some misformed XML in caption; set this option to False for debugging')
    parser.add_argument('-w', '--working_directory', help='Specify the working directory where to read and write files to')
    parser.add_argument('-t', '--testfract', default=0.2, type=float, help='fraction of papers in testset')

    args = parser.parse_args()

    tags2anonymize = args.tags2anonymize
    donotanonymize = args.donotanonymize
    limit = args.limit
    year_range = args.year_range.split(":")
    journals = [j.strip() for j in args.journals.split(",")]
    single_doi = args.doi
    type = args.type
    exclusive_mode = args.exclusive
    keep_roles_only_for_selected_tags = args.selective
    role = args.role
    safe_mode = not args.not_safe_mode

    where_clause= ''
    limit_clause = ''
    entity_type_clause = ''
    entity_role_clause = ''
    tags2anonmymize_clause = ''
    donotanonymize_clause = ''

    if single_doi or year_range[0] or journals[0]:
        where_clause = "WHERE "
        single_doi_clause = ''
        year_range_clause = ''
        journal_clause = ''
        if single_doi: single_doi_clause = "a.doi = '{}' ".format(single_doi)
        if year_range[0]: year_range_clause = "toInteger(a.year) >= {} AND toInteger(a.year) <= {}".format(year_range[0], year_range[1])
        if journals[0]: journal_clause = " OR ".join(["a.journalName =~ '(?i).*{}.*' ".format(j) for j in journals])
        where_clause += " AND ".join(["({})".format(c) for c in [single_doi_clause, year_range_clause, journal_clause] if c])
        print("where_clause", where_clause)

    if limit:
        limit_clause = " LIMIT {} ".format(limit)
        print("limit_clause", limit_clause)

    if type:
        if type == 'entity':
            type = "molecule, gene, protein, subcellular, cell, tissue, organism, undefined"
        type_list = [t.strip() for t in type.split(",")]
        entity_type_clause = ''
        if 'assay' in type_list:
            entity_type_clause += ' t.category = "assay" '
            type_list.remove('assay')
            if type_list: entity_type_clause +='OR t.type = '
        else:
            entity_type_clause += ' t.type = '
        entity_type_clause += " OR t.type = ".join(["'{}'".format(t) for t in type_list])
        entity_type_clause = "AND ({}) ".format(entity_type_clause)
        print("entity_type_clause", entity_type_clause)

    if role:
        entity_role_clause = ' t.role = '
        entity_role_clause += " OR t.role = ".join(["'{}'".format(t.strip()) for t in role.split(",")])
        entity_role_clause = "AND ({}) ".format(entity_role_clause)
        print("entity_role_clause", entity_role_clause )

    if tags2anonymize:
        if tags2anonymize == 'entity':
            tags2anonymize = "molecule, gene, protein, subcellular, cell, tissue, organism, undefined"
        tags2anonymize = ['"{}"'.format(t.strip()) for t in tags2anonymize.split(',')]
        tags2anonmymize_clause = ", ".join(tags2anonymize)
    print("tags2anonmymize_clause", tags2anonmymize_clause)

    if donotanonymize:
        donotanonymize = ['"{}"'.format(t.strip()) for t in donotanonymize.split(',')]
        donotanonymize_clause = ", ".join(donotanonymize)
    print("donotanonymize_clause", donotanonymize_clause)

    options = {}
    options['verbose'] = args.verbose
    options['namebase'] = args.namebase
    options['testset_fraction'] = args.testfract
    options['where_clause'] = where_clause
    options['entity_type_clause'] = entity_type_clause
    options['entity_role_clause'] = entity_role_clause
    options['tags2anonmymize_clause'] = tags2anonmymize_clause
    options['donotanonymize_clause'] = donotanonymize_clause
    options['limit_clause'] = limit_clause
    options['safe_mode'] = safe_mode
    options['exclusive_mode'] = exclusive_mode
    options['keep_roles_only_for_selected_tags'] = keep_roles_only_for_selected_tags
    options['source'] = {'db': 'http://localhost:7474/db/data/', 'username': 'neo4j', 'password': 'sourcedata'} #getpass()}
    
    if options['verbose']: print(options)
    if args.working_directory:
        config.working_directory = args.working_directory

    with cd(config.working_directory):
        G = NeoImport(options)
        errors = G.neo2xml(options['source'])
        G.log_errors(errors)
        trainset, testset = G.split_dataset(options['testset_fraction'])
        G.save({'train': trainset, 'test': testset}, config.data_dir, options['namebase'])

if __name__ == "__main__":
    main()




