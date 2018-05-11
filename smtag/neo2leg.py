import re
from xml.etree.ElementTree import fromstring, Element, SubElement, tostring
from neo4jrestclient.client import GraphDatabase, Node
from random import random
import difflib
#from copy import deepcopy

#DB = GraphDatabase("http://localhost:7474/db/data/",username="neo4j",password="sourcedata")
#DB = GraphDatabase("http://sdtag.net:7474/db/data/",username="neo4j",password="sourcedata")
#amazon elastic IP is 34.202.28.116 and domain is sdtag.net
SD_PANEL_OPEN =  "<sd-panel>"
SD_PANEL_CLOSE = "</sd-panel>"

MARKING_CHAR = u'\uE000'
MARKING_CHAR_ORD = ord(MARKING_CHAR)


#ENTITY_LIST = {}
#with open("entity_list.txt",'r') as f:
#    for l in f: ENTITY_LIST.append(l)
#close(f)


def anonymize_sdtags(panel_xml, tags_neo):
   # mini_index = [] #keeps an index of terms that are the same; will be anonymized by respective 'marker' characters
    for t in tags_neo:
        tag_xml = panel_xml.find('.//sd-tag[@id="sdTag{}"]'.format(t['data']['id']))  
        if tag_xml is not None: 
            #inner_text = ''.join([t for t in tag_xml.itertext()])
            #for sub in tag_xml.iter(): 
            if tag_xml.text:
                #tag_text_lo = tag_xml.text.lower()
                #if tag_text_lo not in mini_index: mini_index.append(tag_text_lo)
                #mark = unichr(MARKING_CHAR_ORD + mini_index.index(tag_text_lo))
                #tag_xml.text = mark * len(tag_xml.text)
                tag_xml.text = MARKING_CHAR * len(tag_xml.text)
                #elif mode == 'randomize':
                # tag_xml.text = ''.join([choice(string.ascii_letters) for _ in tag_xml.text])
                #BETTER IDEA: instead of randomizing text, use random entity from dictionary et Uniprot or ChEBI or SD
                #elif mode == 'random_entities':
                    #if random() < P:
                        #rand_entity = choice(ENTITY_LIST)
                        #tag_xml.text = ran_entity
    return panel_xml

def caption_text2xml(panel_caption, tags, tags2anonym, safe_mode = True, exclusive_mode = False, keep_roles_only_for_selected_tags = False):
    tag_errors = []
    panel_caption = panel_caption.encode('utf-8')
    if safe_mode:
        #need protection agains missing spaces
        
        #protection against carriage return
        if re.search('[\r\n]', panel_caption):
            print "WARNING: removing return characters"
            panel_caption = re.sub('[\r\n]','', panel_caption)
        
        #protection against <br> instead of <br/>
        panel_caption = re.sub(r'<br>', r'<br/>', panel_caption)  
        
        #protection against badly formed link elements
        panel_caption = re.sub(r'<link href="(.*)">', r'<link href="\1"/>', panel_caption)                
        panel_caption = re.sub(r'<link href="(.*)"/>(\n|.)*</link>', r'<link href="\1">\2</link>', panel_caption)
        #protection agains missing <sd-panel> tags
        if re.search(r'^{}(\n|.)*{}$'.format(SD_PANEL_OPEN, SD_PANEL_CLOSE), panel_caption) is None:
            print "WARNING: correcting missing <sd-panel> </sd-panel> tags!"
            print panel_caption
            panel_caption = SD_PANEL_OPEN + panel_caption + SD_PANEL_CLOSE
        

    #We may loose a space that separates panels in the actual figure legend... 
    panel_caption = re.sub('</sd-panel>$', ' </sd-panel>', panel_caption)
    #and then remove possible runs of spaces
    panel_caption = re.sub(r' +',r' ', panel_caption)
    
    panel_xml = fromstring(panel_caption)
    #original_panel_xml = deepcopy(panel_xml)
    
    tags_xml = panel_xml.findall('.//sd-tag')
    tags_neo_id = [u"sdTag{}".format(t['data']['id']) for t in tags]
    tags_not_found = set(tags_neo_id) - set([t.attrib['id'] for t in tags_xml])
    if tags_not_found:
        print "WARNING, tag(s) not found: ", tags_not_found
        print panel_caption
        tag_errors.append(tags_not_found)
    
    #keep attributes only for the selected tags and clear the rest 
    if exclusive_mode:
        for t_xml in tags_xml:
            if 'id' in t_xml.attrib: 
                if not t_xml.attrib['id'] in tags_neo_id: 
                    t_xml.attrib.clear()
            else:
                print "WARNING, tag", tostring(t_xml), "has no id" 
    
    if keep_roles_only_for_selected_tags:
        for t_xml in tags_xml:
            if 'id' in t_xml.attrib and 'role' in  t_xml.attrib:
                if not t_xml.attrib['id'] in tags_neo_id:
                    t_xml.attrib.pop('role')
                        
    #anonymize a subset of the tags
    if tags2anonym:
        anonymize_sdtags(panel_xml, tags2anonym)
        
    #if tags2augment:
    #    panels_xmls = augment(panel_xml, tags2augment, augmentation_factor)
    #should return an array of panel_xmls in case of data augmentation at this stage
    return panel_xml, tag_errors #, original_panel_xml
        
def neo2xml(source, options):
    
    where_clause = options['where_clause']
    entity_type_clause = options['entity_type_clause']
    entity_role_clause = options['entity_role_clause']
    tags2anonmymize_clause = options['tags2anonmymize_clause']
    donotanonymize_clause = options['donotanonymize_clause']
    limit_clause = options['limit_clause']
    safe_mode = options['safe_mode']
    exclusive_mode = options['exclusive_mode']
    keep_roles_only_for_selected_tags = options['keep_roles_only_for_selected_tags']
    
    DB = GraphDatabase(source['db'],source['username'], source['password'])
    figure_captions_xml = {}
    #figure_captions_text = {}
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
    
    for a in results_articles:
        a_id = a[0]
        doi = a[1]
        if doi == '': doi = a_id
        if doi in figure_captions_xml:
            print 'WARNING! {} ALREADY EXISTS'.format(doi)
            paper_errors.append({a_id, doi})
        else:
			q_figures = '''
				MATCH (a:Article )-->(f:Figure)
				WHERE id(a) = {}
				RETURN id(f), f.fig_label, f.caption
				ORDER BY f.fig_label ASC
				'''.format(a_id)
			results_figures = DB.query(q_figures)
		
			figure_captions_xml[doi]= []
			#figure_captions_text[doi] = []
		
			for f in results_figures:
				f_id = f[0]
				fig_label = f[1]
				fig_original_caption = f[2].encode('utf-8')
				#fig_original_caption = cleanup(fig_original_caption)
			  
				q_panel = '''
				   MATCH (f:Figure)-->(p:Panel)-->(t:Tag)
				   WHERE id(f) = {} AND t.in_caption = true
				   {} //AND t.type = some_entity_type OR some other type
				   {} //AND t.role = some role OR some role
				   WITH p.formatted_caption AS formatted_caption, p.label AS label, p.panel_id AS panel_id, COLLECT(DISTINCT t) AS tags
				   RETURN formatted_caption, label, panel_id, tags , [t in tags WHERE (t.type in [{}] AND NOT t.role in[{}])] AS tags2anonym // (t.type in ["gene","protein"] AND NOT t.role in ["reporter"])
			   
				  '''.format(f_id, entity_type_clause, entity_role_clause, tags2anonmymize_clause, donotanonymize_clause)
				results_panels = DB.query(q_panel)
				#print "querying with:"
				#print q_panel
				print (u"{} panels found for figure {} ({}) in paper {}".format(len(results_panels), fig_label, f_id, doi)).encode('utf-8')
			
				if results_panels:              
					figure_xml_element = Element('figure-caption')
					#figure_original_text = ''
					#panels not in the proper order, need resorting via label
					results_labeled = {p[1]:{'panel_caption':p[0], 'panel_id':p[2], 'fig_label':fig_label, 'tags':p[3], 'tags2anonym':p[4]} for p in results_panels}
					sorted_panel_labels = results_labeled.keys()
					sorted_panel_labels.sort()
				
					for p in sorted_panel_labels:      
						panel_caption = results_labeled[p]['panel_caption']
						tags = results_labeled[p]['tags']
						tags2anonym = results_labeled[p]['tags2anonym']
						try:
							panel_xml_element, tag_errors = caption_text2xml(panel_caption, tags, tags2anonym, safe_mode, exclusive_mode, keep_roles_only_for_selected_tags)
							#generate multiple data augmented figures
							#for i in range(len(figure_xml_element_list)): figure_xml_element_list[i].append(panel_xml_element[i])
							figure_xml_element.append(panel_xml_element)
							#figure_original_text = figure_original_text + ''.join([t for t in panel_xml_element.itertext()])
							if tag_errors:
								panel_id = results_labeled[p]['panel_id']
								fig_label = results_labeled[p]['fig_label']
								tag_level_errors.append([doi, fig_label, p, panel_id, panel_caption, tag_errors])
						except Exception as e:
							panel_id = results_labeled[p]['panel_id']
							fig_label = results_labeled[p]['fig_label']
							print (u"problem parsing fig {} panel {} (panel_id:{}) in article {}".format(fig_label, p, panel_id, doi)).encode('utf-8')
							print panel_caption.encode('utf-8')
							print " ==> error: ", e, "\n"
							caption_errors.append([doi, fig_label, p, panel_id, e, panel_caption])
				
					#for f in figure_xml_element_list: figure_captions[a_id].append(f)
					figure_captions_xml[doi].append(figure_xml_element)
				
					#figure_captions_text[doi].append(figure_original_text)
				
					#cleanup xml for missing spaces
					#panel_inner_text = ''.join([s for s in figure_xml_element.itertext()])
					#fig_original_caption = "<fig>{}</fig>".format(fig_original_caption)
					#original_inner_text = ''.join([s for s in fromstring(fig_original_caption).itertext()])
					#print "\n\n\npanel_inner_text:\n"
					#print panel_inner_text
					#print "\n\n\noriginal_inner_text:\n"
					#print original_inner_text
				
			print "number of figures in ", a_id, doi, len(figure_captions_xml[doi])
    return figure_captions_xml, {'paper_level':paper_errors, 'caption_level': caption_errors, 'tag_level': tag_level_errors} #figure_captions_text
    
 