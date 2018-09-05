# -*- coding: utf-8 -*-
#T. Lemberger, 2018

class XGraph():

    @staticmethod
    def xml2adj(element, adj, index):
        if element is not None:
            tag = element.tag
            if tag == 'sd-tag':
                text = cleanup(element.text)
                category = element.attrib['category']
                if category =='':
                    e_type = element.attrib['type']
                    e_role = element.attrib['role']
                    i = index.add(text, e_type)
                    adj.update(i, e_role)

            #add marks recursively
            for child in list(element):
                XGrap.xml2adj(child, adj, index)
