import argparse
from getpass import getpass
from dataprep import DataPreparator
from featurizer import XMLFeaturizer
import neo2leg

class SDGraphPreparator(DataPreparator):

    def __init__(self, parser):
        super(SDGraphPreparator, self).__init__(parser) #hopefully parser is mutable otherwise use self.parser
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
                
        self.options = self.set_options(parser.parse_args())
        if self.options['verbose']: print self.options
        super(SDGraphPreparator, self).main()
        
    @staticmethod
    #implements @abstractmethod
    def set_options(args):
        options = super(SDGraphPreparator, SDGraphPreparator).set_options(args)
        
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
            print "where_clause", where_clause

        if limit:    
            limit_clause = " LIMIT {} ".format(limit)
            print "limit_clause", limit_clause

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
            print "entity_type_clause", entity_type_clause

        if role: 
            entity_role_clause = ' t.role = '
            entity_role_clause += " OR t.role = ".join(["'{}'".format(t.strip()) for t in role.split(",")])
            entity_role_clause = "AND ({}) ".format(entity_role_clause)
            print "entity_role_clause", entity_role_clause 

        #, [t in tags WHERE t in ["protein","gene"]] AS tags_to_anonymize   
        if tags2anonymize: 
            if tags2anonymize == 'entity':
                tags2anonymize = "molecule, gene, protein, subcellular, cell, tissue, organism, undefined"
            tags2anonymize = ['"{}"'.format(t.strip()) for t in tags2anonymize.split(',')]
            tags2anonmymize_clause = ", ".join(tags2anonymize)
        print "tags2anonmymize_clause", tags2anonmymize_clause

        if donotanonymize: 
            donotanonymize = ['"{}"'.format(t.strip()) for t in donotanonymize.split(',')]
            donotanonymize_clause = ", ".join(donotanonymize)
        print "donotanonymize_clause", donotanonymize_clause
        
        options['where_clause'] = where_clause
        options['entity_type_clause'] = entity_type_clause
        options['entity_role_clause'] = entity_role_clause
        options['tags2anonmymize_clause'] = tags2anonmymize_clause
        options['donotanonymize_clause'] = donotanonymize_clause
        options['limit_clause'] = limit_clause
        options['safe_mode'] = safe_mode
        options['exclusive_mode'] = exclusive_mode
        options['keep_roles_only_for_selected_tags'] = keep_roles_only_for_selected_tags
        options['source'] = {'db': 'http://localhost:7474/db/data/', 'username': 'neo4j', 'password': getpass()}
        
        return options

    #implements @abstractmethod
    def build_feature_dataset(self, xml_papers):
    
        dataset = []

        for id in xml_papers:
            #print "Paper: ", id, xml_papers[id]
            for i in range(len(xml_papers[id])):        
                figure_xml = xml_papers[id][i]
                text = ''.join([s for s in figure_xml.itertext()])
                features, _, _ = XMLFeaturizer.xml2features(figure_xml)

                if text:
                    dataset.append({'provenance':{'id':id,'index':i+1}, 'text': text,'features': features})
                else:
                    print "skipping an example in paper with id=", id
                    print "<xml>{}</xml>".format(tostring(figure_xml))
                    print

        return dataset
    
    #implements @abstractmethod
    def import_examples(self, source):
        raw_examples = [] #where each raw_example i is e[i]['text']  e[i]['provenance'] and opt e[i]['ann']
        #neo2xml(where_clause = '', entity_type_clause = '', entity_role_clause = '', tags2anonmymize_clause = '', donotanonymize_clause = '', limit_clause = '', safe_mode = True, exclusive_mode = False, keep_roles_only_for_selected_tags = False)
        raw_examples, errors = neo2leg.neo2xml(source, self.options)
        return raw_examples
        
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Generates text and tensor files from tagged text in sd-graph.")
    p = SDGraphPreparator(parser)
