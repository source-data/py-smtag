import re
from typing import List
from argparse import ArgumentParser
from xml.etree.ElementTree import Element, fromstring, tostring, ParseError


def innertext(element: Element) -> str:
    return "".join([t for t in element.itertext()])

def special_innertext(element:Element, add_after =['.//sd-panel/b[1]'], add_before=['.//sd-panel']) -> str:
    def add_space_after(element: Element):
        for xpath in add_after:
            for e in element.findall(xpath):
                if e.tail is None:
                    e.tail = ' '
                elif e.tail[0] != ' ':
                    e.tail = ' ' + e.tail

    def add_space_before(element: Element):
        for xpath in add_before:
            for e in element.findall(xpath):
                for sub in e[::-1]:
                    if sub.tail is not None:
                        if sub.tail[-1] != ' ':
                            sub.tail += ' '
                        break

    # def remove_double_spaces(element: Element):
    #     s = tostring(element, encoding='unicode')
    #     replaced = re.sub(r' ((?:<[^>]+>)+) ', r' \1', s)
    #     try:
    #         new_xml = fromstring(replaced)
    #     except ParseError as err:
    #         # junk after document element: line 1, column 441
    #         print("PARSING ERROR IN:")
    #         print(replaced)
    #         print()
    #         column = int(re.search(r'column (\d+)', str(err)).group(1))
    #         print('culprit:')
    #         print(replaced[column])
    #         print('=================')
    #         raise err
    #     return new_xml

    # before taking the innertext we nee to make sure we remove any tail so that only the inner conent is considered
    element.tail = None 
    add_space_after(element)
    add_space_before(element)
    new_xml = element #remove_double_spaces(element)
    inner_text = innertext(new_xml)
    return inner_text, new_xml

def main():
    test_text = '''<fig><sd-panel><b>(B)</b><sd-tag external_database0="NCBI gene" external_id0="110213" id="sdTag272" role="component" type="gene">BI‐1</sd-tag> KO <sd-tag external_database0="CVCL" external_id0="CVCL_9115" id="sdTag273" role="component" type="cell">MEFs</sd-tag> were stably transduced with lentiviral vectors as described in <b>A</b>. Cells were exposed to <sd-tag id="sdTag274" role="component" type="undefined">EBSS</sd-tag> for 3 h, and then <sd-tag external_database0="Uniprot" external_database1="Uniprot" external_id0="Q91VR7" external_id1="Q9CQV6" id="sdTag276" role="assayed" type="protein">LC3</sd-tag> levels were analysed by <sd-tag category="assay" external_database0="BAO" external_id0="BAO_0002424" id="sdTag277">western blot</sd-tag>. Image was assembled from cropped lanes of the same <sd-tag category="assay" external_database0="BAO" external_id0="BAO_0002424" id="sdTag278">western blot</sd-tag> analysis.<graphic href="https://api.sourcedata.io/file.php?panel_id=5247" /></sd-panel><sd-panel><b>(C)</b> Endogenous <sd-tag external_database0="Uniprot" external_database1="Uniprot" external_id0="Q91VR7" external_id1="Q9CQV6" id="sdTag281" role="assayed" type="protein">LC3</sd-tag> distribution was visualized using immunofluorescence and <sd-tag category="assay" external_database0="BAO" external_id0="BAO_0000453" id="sdTag283">confocal microscopy</sd-tag> in <sd-tag external_database0="NCBI gene" external_id0="110213" id="sdTag284" role="component" type="gene">BI‐1</sd-tag> KO/shLuc and <sd-tag external_database0="NCBI gene" external_id0="110213" id="sdTag285" role="component" type="gene">BI‐1</sd-tag> KO/sh<sd-tag external_database0="NCBI gene" external_id0="56208" id="sdTag286" role="intervention" type="gene">Beclin‐1</sd-tag> cells. Quantification represents the visualization of at least 180 cells. Student's <i>t</i>‐test was used to analyse statistical significance. Mean and standard deviation are presented, <sup>*</sup><i>P</i>0.001, NS: non‐significant. <graphic href="https://api.sourcedata.io/file.php?panel_id=5248" /></sd-panel><sd-panel><b>(D)</b><sd-tag external_database0="Uniprot" external_database1="Uniprot" external_id0="Q91VR7" external_id1="Q9CQV6" id="sdTag290" role="assayed" type="protein">LC3</sd-tag> was visualized and quantified in <sd-tag external_database0="NCBI gene" external_id0="110213" id="sdTag291" role="component" type="gene">BI‐1</sd-tag> KO/shLuc and <sd-tag external_database0="NCBI gene" external_id0="110213" id="sdTag292" role="component" type="gene">BI‐1</sd-tag> KO/sh<sd-tag external_database0="NCBI gene" external_id0="78943" id="sdTag299" role="intervention" type="gene">IRE1α</sd-tag> cells described in (<b>B</b>) by immunofluorescence and <sd-tag category="assay" external_database0="BAO" external_id0="BAO_0000453" id="sdTag294">confocal microscopy</sd-tag> analysis. <graphic href="https://api.sourcedata.io/file.php?panel_id=5249" /></sd-panel></fig>'''

    argparse = ArgumentParser(description="space keeping innertext")
    argparse.add_argument('xml_string', nargs="?", default=test_text, type=str, help="the xml string to process")
    args = argparse.parse_args()
    xml_string = args.xml_string
    xml = fromstring(xml_string)
    inner_text, new_xml = special_innertext(xml)
    print(inner_text)

if __name__ == "__main__":
    main()
