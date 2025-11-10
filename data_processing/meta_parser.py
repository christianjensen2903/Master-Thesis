# -.- coding: utf-8 -.-
__author__ = "dcg601"

from urllib.parse import urlparse, parse_qs

from unidecode import unidecode

"""
Module to read the metadata XML and extract certain values
"""

import html
import os

from lxml import etree  # type: ignore
from io import StringIO
import json

import re
import logging
from collections import OrderedDict
import os.path as path
from bs4 import BeautifulSoup as bs
import glob
import lxml
import sys
import pandas as pd  # type: ignore
import fnmatch

logger = logging.getLogger(__name__)

# Set logger to info level
logger.setLevel(logging.WARNING)


CELEX_REG = r"6(?P<year>\d{4})(?P<court>\w)(?P<type>\w)(?P<no>\d{4})"
CASE_REG = r"(((?P<court>[CT]).)?(?P<no>\d{1,3})/(?P<year>\d{1,2}))"


def celex_to_case_dict(celex):
    """
    Decomposes CELEX and returns a dictionary of
    court letter, case number, year
    """
    matcher = re.search(re.compile(CELEX_REG), celex)

    if matcher:
        year = matcher.group("year")
        court = matcher.group("court")
        no = str(int(matcher.group("no")))

        return {
            "court": court,
            "number": no,
            "year": year,
        }

    else:
        raise ValueError


def celex_to_caseid(celex):
    """
    Given a celex number it returns it case id
    Does not return extra letters at the end eg "C-200/12 P"
    """
    d = celex_to_case_dict(celex)

    # the case id comprises of court letter,
    # dash, case number
    # slash, 2-last year digits
    id = d["court"] + "-" + d["number"] + "/" + d["year"][-2:]
    return id


def caseid_to_celex(caseid):
    """
    Given the case id return the celex
    Parameters
    ----------
    caseid : str
        Case id of the form C-145/90 or 145/90

    Returns
    -------
    celex : str
        The celex number of the case
    """

    m = re.search(CASE_REG, caseid, re.I)

    if m:
        if m.group("court"):
            court = m.group("court")
        else:
            court = "C"
        number = int(m.group("no"))
        year = int(m.group("year"))

        year = year + 1900 if (year <= 99) and (year > 50) else year + 2000

        return "6%(year)d%(court)sJ%(number)04d" % {
            "year": year,
            "court": court,
            "number": number,
        }
    else:
        return None


def walker(pattern, root=os.curdir):
    """Locate all files matching supplied filename pattern in and below
    supplied root directory."""
    for path, dirs, files in os.walk(os.path.abspath(root)):
        for filename in fnmatch.filter(files, pattern):
            yield os.path.join(path, filename)


class Reader(object):
    """
    Reads specific metadata from the XML metafiles. The cases are read from the root directory and the metadata is
    output in batches of size `batch_size` at the `output_dir`. Cases are read read in order of modification time,
    ie time they were downloaded.
    """

    def __init__(self, root, batch_size=500):
        """
        Init with a root folder, where all cases are contained in subdirectories
        :param root:
        :return:
        """
        self._root = root
        self.output = OrderedDict()
        self._batch_size = batch_size

        sys.stderr.write("glob\n")
        # Get all folders in the root directory
        directories = []
        for item in os.listdir(self._root):
            item_path = path.join(self._root, item)
            if path.isdir(item_path):
                directories.append(item_path)

        # Sort directories by modification time
        directories = sorted(directories, key=os.path.getmtime)

        xmlfiles = []
        logger.info("Iterating judgment directories for metadata files")
        for d in directories:

            # Skip if not CJ judgment
            if "CJ" not in d:
                continue

            # Try English metadata first
            metafile = path.join(d, "eng_metadata.xml")

            if not path.exists(metafile):
                metafile = path.join(d, "metadata.xml")

            if not path.exists(metafile):
                metafile = path.join(d, "fra_metadata.xml")

            if path.exists(metafile):
                xmlfiles.append(metafile)
            else:
                logger.warning("No metadata file found in %s" % d)

        self.xmlfiles = xmlfiles

        # languages instead of the CSV
        self._languages = """"lang2","lang3"
"DE","DEU"
"SV","SWE"
"FI","FIN"
"PT","POR"
"BG","BUL"
"EL","ELL"
"MT","MLT"
"LT","LIT"
"EN","ENG"
"LV","LAV"
"IT","ITA"
"FR","FRA"
"HU","HUN"
"ES","SPA"
"ET","EST"
"CS","CES"
"SK","SLK"
"SL","SLV"
"PL","POL"
"DA","DAN"
"RO","RON"
"NL","NLD"
"HR","HRV"
"GA","GLE"
"""

    def _get_language_from_path(self, filepath):
        # Extract language from filename: eng_metadata.xml -> ENG, fra_metadata.xml -> FRA
        filename = path.basename(filepath)
        if filename.startswith("eng_"):
            return "ENG"
        elif filename.startswith("fra_"):
            return "FRA"
        else:
            # Fallback: try to infer from filename
            logger.warning(f"Could not infer language from filename: {filename}")
            return None

    def _get_language_2(self, lang3):
        """
        Get the language 2-letter acronym giving as input the 3letter acro
        Parameters
        ----------
        lang3:
            The 3 letter acronym, e.g. ENG

        Returns
        -------
        The 2 letter acronym eg. EN
        """

        df = pd.read_csv(StringIO(self._languages), delimiter=",", header=0)
        try:
            lang2 = df[df["lang3"] == lang3]["lang2"].values[0]
        except IndexError:
            sys.stderr.write(f"Language {lang3} not found in the list of languages\n")
            lang2 = None
            exit(-1)
        return lang2

    def _get_celex_from_path(self, filepath):
        # CELEX is the directory name containing the metadata file
        celex = path.basename(path.dirname(filepath))
        return celex

    def _sort_json(self, js: dict):
        """
        sort the json values (of the global metadata file according to decision date, because they can be
        in rather random order
        :param js: the dictionary with the global metadata
        :return: sorted js
        """

        jsarr = [(it[0], it[1]) for it in js.items()]
        jsarr.sort(key=lambda x: x[1]["meta"]["date"])
        js = {ar[0]: ar[1] for ar in jsarr}
        return js

    def infer_filenames(self, meta_filename, case_lang3):
        """
        Heuristic to infer the available files and language versions given the files with the metadata

        :param meta_filename: The name of the XML file given as input
        :param case_lang3: The language of the case
        :return: dictionary with files and language versions
        """

        case_dir = path.dirname(meta_filename)
        # case_dir is now the judgment/CELEX directory directly
        dir_celex = case_dir
        logger.info(f"--- Inferring filenames for {path.basename(dir_celex)}")

        case_lang = self._get_language_2(case_lang3)
        logger.info("Language of the case with 2 letters:%s" % case_lang)

        files_to_return = {}
        # Map language codes to filename prefixes
        lang_prefix_map = {"EN": "eng", "FR": "fra"}
        if case_lang:
            # Get 3-letter code for case language to determine prefix
            case_lang3_lower = case_lang3.lower() if case_lang3 else None
            if case_lang3_lower:
                lang_prefix_map[case_lang] = case_lang3_lower[:3]

        # are available. We hope that at least English or French is available together with the case language
        for lang in {"EN", "FR", case_lang}:
            if lang is None:
                continue
            lang_prefix = lang_prefix_map.get(lang, lang.lower()[:3])
            # Try both naming conventions: eng_judgment.html and judgment.html
            filename = path.join(dir_celex, f"{lang_prefix}_judgment.html")
            if not path.exists(filename):
                continue

            files_to_return[lang] = {
                "judgment": filename,
            }

        return files_to_return

    def values_from_xml(self, file):

        # try:
        parser = EcjXmlParser(file)
        # except NoMetadataError:
        #     return None

        celex = parser.read_celex()
        logger.debug("CELEX:%s" % celex)
        citations = parser.read_citations_and_citing_paragraphs()
        logger.debug("Citations:%s" % str(citations))

        if len(citations) == 0:
            logger.info("No citations found for %s." % celex)
            case_dir = path.dirname(file)
            # Try English HTML first, then French, then fallback to simple judgment.html
            html_case_file = path.join(case_dir, "eng_judgment.html")
            if not os.path.exists(html_case_file):
                logger.info(
                    "English HTML version doesn't exist, will try to extract citations from the French"
                )
                html_case_file = path.join(case_dir, "fra_judgment.html")
            if not os.path.exists(html_case_file):
                logger.info(f"No HTML file found in {case_dir}")
            else:
                html_parser = EcjHtmlParser(html_case_file)
                citations = html_parser.read_citations_and_citing_paragraphs()
                logger.debug("Citations from HTML:%s" % str(citations))

        # output = OrderedDict()
        output = {}
        output["CELEX"] = celex

        # AG Opinions don't have a language of the case as decisions/judgments
        authentic_lang = (
            parser.read_language_of_case() if "CC" not in celex else ["ENG"]
        )
        logger.info("Authentic language of the case: %s" % authentic_lang)
        if len(authentic_lang) > 0:
            output["files"] = self.infer_filenames(file, authentic_lang[0])
        else:
            logger.error("not possible to detect original language in %s" % celex)
            output["files"] = {}
        output["meta"] = {}  # OrderedDict()
        output["meta"]["ECLI"] = parser.read_ecli()
        output["meta"]["caseno"] = parser.read_caseno(celex)
        output["meta"]["full_title"] = parser.read_full_title()
        output["meta"]["short_title"] = parser.read_short_title()
        output["meta"]["references"] = citations
        try:
            output["meta"]["application_date"] = parser.read_application_date()[0]
        except IndexError:
            # In some less important orders, the application date is not filled in
            output["meta"]["application_date"] = None
        output["meta"]["date"] = parser.read_document_date()[0]
        # language of the case can have multiple values
        output["meta"]["authentic_language"] = parser.read_language_of_case()
        try:
            output["meta"]["subject_matter"] = parser.read_subject_matter()
        except lxml.etree.XPathEvalError:
            logger.exception("Xpath Exception for Subject matter on %s" % celex)
            raise Exception("Xpath Exception on %s" % celex)

        output["meta"]["case_law_about"] = parser.read_case_law_is_about()
        output["meta"]["advocate_general"] = parser.read_advocate_general()
        output["meta"]["rapporteur"] = parser.read_rapporteur()
        output["meta"]["court"] = parser.read_court()

        output["meta"]["document_type"] = parser.read_document_type()
        output["meta"]["procedure_type"] = parser.read_procedure_type()
        output["meta"]["keywords"] = parser.read_keywords()
        output["meta"]["applicant"] = parser.read_applicant()
        output["meta"]["defendant"] = parser.read_defendant()
        observations = parser.read_observations()
        if len(observations) != 0:
            output["meta"]["observations"] = observations
        national_court = parser.read_national_court()
        if national_court is not None:
            output["meta"]["national_court"] = national_court

        del parser

        return output

    def to_file(self, output_dir: str, output: OrderedDict, cur: int):

        # Make the file numbering start from the current-batchsize except when
        # we have a number not divisible by batch_size.
        # Example: if `cur` = 14772, rest is 72, so we need to have a file named `results-14700-14772.json`
        if cur % self._batch_size != 0:
            rest = cur
            start = cur - (cur % self._batch_size)
        else:
            rest = cur
            start = cur - self._batch_size

        os.makedirs(output_dir, exist_ok=True)
        fname = os.path.join(output_dir, "results_%d-%d.json" % (start, rest))
        with open(fname, mode="w+", encoding="utf8") as outfile:
            json.dump(output, outfile, indent=4)
        logger.info(f"Wrote {fname}")
        return

    def add_values(self, output, output_small):
        """
        Gets values from the small directory and populates the larger as output
        """
        celex = output_small["CELEX"]
        output[celex] = {"files": {}, "meta": {}}
        output[celex]["files"] = output_small["files"]
        output[celex]["meta"] = output_small["meta"]

    def process(self, output_dir: str, start_from=0):
        """
        Processes all files that are under the root specified in the constructor and outputs batches of results at
        `output_dir`
        :param output_dir: str the output directory
        :param start_from: int the metafile to start processing. Default value = 0 (the first)
        """
        xmlfiles = self.xmlfiles
        logger.info(f"Will parse {len(xmlfiles)} files")

        batch_size = self._batch_size
        # batch_stops = xmlfiles[::batch_size]
        # start_from_batch = start_from // batch_size

        cur = start_from

        output = OrderedDict()
        for elem in xmlfiles[cur:]:
            logger.info("Parsing %s" % elem)
            if not path.exists(elem):
                logger.info("\tXML not available")
                continue
            try:
                output_small = self.values_from_xml(elem)
                self.add_values(output, output_small)
            except NoMetadataError:
                # Even though the file is present, we can have downloaded a simple placeholder that gives an error
                # message and we therefore throw an exception
                logger.error(f"No metadata for {elem}. Continuing to next result")
                continue

            if (
                cur + 1
            ) % batch_size == 0:  # Every `batch_size` files, output to file and reset the output dictionary
                self.to_file(output_dir, output, cur + 1)
                output = (
                    OrderedDict()
                )  # reinitialize output on every loop so that new meta are output every time

            # increment cur
            cur += 1

        self.to_file(output_dir, output, cur)

        return

    def produce_case_meta(self, meta_root: str, output_json: str):
        """
        Produces the entire case meta file reading the JSON files at meta_root.

        :param meta_root: str
            the root path containing the meta JSON files. Generally, those are outputs of `build_json.py`
        :param output_json: str
            the filename for the JSON with all data
        """
        json_files = glob.glob(os.path.join(meta_root, "*.json"))
        # Try to sort by modification time otherwise the latest meta files can come first
        json_files.sort(key=os.path.getmtime)
        try:
            # Remove from json_files all names starting with 'meta_all'
            json_files = [f for f in json_files if not "meta_all" in f]
            logger.info("removed previous results")
        except ValueError as e:
            pass

        logger.info("Constructing the Global JSON!")

        # construct a global big JSON with all meta
        global_js = {}
        for jsf in json_files:
            logger.info(f"Reading {jsf}")
            json_part = json.load(open(jsf, encoding="utf-8"))
            global_js.update(json_part)
        global_js = self._sort_json(global_js)

        logger.debug("First item:" + list(global_js.keys())[0])
        logger.info("Will write global JSON")

        # and store at the filesystem

        json_out_path = os.path.join(meta_root, output_json)
        with open(json_out_path, mode="w+", encoding="utf-8") as outf:
            json.dump(global_js, outf, indent=4)

        logger.info("Global JSON written successfully!")
        return json_out_path

    def merge_result_batches(
        self, results_dir: str, output_json: str = "all_results.json"
    ) -> str:
        """
        Merges all result batch files (results_*.json) into a single file.

        :param results_dir: str
            the directory containing the result batch JSON files
        :param output_json: str
            the filename for the merged JSON output. Default: "all_results.json"
        :return: str
            the path to the merged output file
        """
        json_files = glob.glob(os.path.join(results_dir, "results_*.json"))
        # Sort by filename to ensure proper order (results_0-500.json, results_500-1000.json, etc.)
        json_files.sort()

        # Exclude the output file if it already exists
        output_path = os.path.join(results_dir, output_json)
        json_files = [f for f in json_files if f != output_path]

        logger.info(f"Found {len(json_files)} batch files to merge")

        # construct a global big JSON with all results
        global_js = OrderedDict()
        total_cases = 0

        for jsf in json_files:
            logger.info(f"Reading {os.path.basename(jsf)}")
            json_part = json.load(open(jsf, encoding="utf-8"))
            batch_size = len(json_part)
            total_cases += batch_size
            global_js.update(json_part)
            logger.info(f"  Added {batch_size} cases (total: {total_cases})")

        logger.info(f"Merged {total_cases} cases total")
        global_js = self._sort_json(global_js)

        logger.info("Will write merged results JSON")

        # and store at the filesystem
        with open(output_path, mode="w+", encoding="utf-8") as outf:
            json.dump(global_js, outf, indent=4, ensure_ascii=False)

        logger.info(f"Merged results JSON written successfully to {output_path}!")
        return output_path


class NoMetadataError(Exception):
    pass


def is_allowed_document_type(celex: str) -> bool:

    # Define allowed combinations: (section, letters)
    allowed_combinations = {
        ("6", "CJ"),  # Court of Justice - Judgment
        ("3", "L"),  # Directives
        ("3", "R"),  # Regulations
    }

    # Pattern to extract section and letters
    # Sector 6: 6YYYY[Court][Type]####
    # Sector 3: 3YYYY[Type]####
    pattern = re.compile(r"^(\d)\d{4}([A-Z]{1,2})\d{4}$")
    match = pattern.match(celex)

    if match:
        section = match.group(1)
        letters = match.group(2)
        return (section, letters) in allowed_combinations

    return False


class EcjHtmlParser(object):
    """
    Parses HTML files and extracts the citations to other cases and to relevant paragraphs if those citations exist
    """

    def __init__(self, filepath):
        self.filepath = filepath
        logger.info(f"Will use HTML parser to parse citations from: {filepath}")

        # Isolate the name of the case from the path by finding the name
        # of filepath's parent (the judgment/CELEX folder)
        self.case_name = path.basename(path.dirname(filepath))

    def _isolate_uri(self, url: str):
        """
        Isolates the URI from the URL passed as string
        """
        parsed_url = urlparse(url)
        params = parse_qs(parsed_url.query)
        uri = params["uri"][0]

        return uri

    def __year_from_caseno(self, caseno: str):
        """
        Extracts the year from the case number
        """
        # Hack: extract the last 2 caseno digits as case numbers
        # are of the form C-123/12
        return re.split(r"\s|,", caseno)[0][-2:]

    def _extract_paragraph_number(self, link):
        """
        Extracts the paragraph number from the link element which is inside a paragraph element
        """
        logger.info(f"\tLink is {link} with text {link.text}")
        paragraph = link.parent
        paragraph_number = None
        # The paragraph number is the text of a grandparent of the link with class "coj-count"
        while True:
            paragraph_number_element = paragraph.find_previous(
                "p", attrs={"id": re.compile(r"point\d+")}
            )
            if paragraph_number_element is not None and str.isnumeric(
                paragraph_number_element.get_text(strip=True)
            ):
                # In some cases what is detected as paragraph number is actually a number in parentheses or a letter
                # in which cases we need to check if the string is numeric, otherwise this is not the real
                # paragraph number
                paragraph_number = paragraph_number_element.get_text(strip=True)
                logger.info("\tParagraph number: %s" % paragraph_number)
                paragraph = paragraph.parent
                break
            else:
                paragraph = paragraph.parent
                if paragraph is None or paragraph.get_text(strip=True) == "–":
                    logger.debug(
                        "\tReached the top or found a dash in the paragraph start"
                    )
                    break
                logger.info(
                    "\tParagraph number not found yet, advancing one step upwards"
                )
        # if paragraph_number contains is a number in () return None else return paragraph_number
        try:
            if not re.match(r"\(\d+\)", paragraph_number):
                return paragraph_number
            else:
                return None
        except TypeError:
            sys.exit(-1)

    def _extract_paragraph_ranges(self, case_links, number_links):
        """
        The method extracts paragraphs and paragraph ranges in the judgment.
        :param case_links: list
            The URLs of the case links
        :param number_links: list
            The URLs of the paragraph links (that's why we call it number_links)
        :return:
            links_with_ranges: list
                the links with the paragraph ranges
        """
        logger.info(f"\tEntering _extract_paragraph_ranges for {self.filepath}")
        links_with_ranges = []

        i = 0

        if len(number_links) == 0:
            # Will return a list with only the links to the cases and None as the target paragraph
            logger.info("No paragraph links found. Will return only the case links")
            return [(l, None) for l in case_links]

        for l in case_links:
            # Iterate over case_links and number_links and try to detect non-matching hrefs and ranges
            # We use the counter i to iterate over the number_links, because we need sometimes to advance by 2 entries
            # in the number_links list, sometimes by 1, sometimes not at all, if the case_link does not
            # have a corresponding paragraph link
            case_link = l
            logger.debug(f"Case link: {unidecode(case_link.get_text(strip=True))}")
            case_year = self.__year_from_caseno(case_link.get_text(strip=True))
            logger.debug(f"Case year: {case_year}")
            if i <= len(number_links) - 1:
                paragraph = number_links[i]
                logger.debug(f"Paragraph link: {paragraph.get_text(strip=True)}")
            else:
                logger.info(
                    "No more paragraph links to process. The remaining links should be case only "
                    "and will be added as case links"
                )
                links_with_ranges.append((l, None))
                continue

            try:
                if self._isolate_uri(case_link["href"]) == self._isolate_uri(
                    paragraph["href"]
                ) and int(case_year) not in range(54, 75):

                    # The case_link and the paragraph link are the same and thus
                    # the case_link IS a paragraph link or we have citation to a very old case without paragraphs
                    logger.info(10 * "=")
                    logger.info(
                        "The case link is a paragraph link. Will search for the sibling of it to detect ranges"
                    )
                    # search if the sibling of paragraph is the text 'to'
                    sibling = paragraph.next_sibling
                    if sibling and sibling.string.strip() == "to":
                        logger.info(
                            "The sibling of the paragraph link is 'to'. Will extract the range"
                        )
                        # The citation is citation to paragraph range
                        target_number = int(sibling.next_sibling.get_text(strip=True))
                        cited_range = range(
                            int(paragraph.get_text()), int(target_number + 1)
                        )
                        logger.debug(f"Extracted range: {list(cited_range)}")
                        links_with_ranges.append((l, [str(p) for p in cited_range]))
                        i += 2
                    elif (
                        sibling
                        and (sibling.string.strip() == "and")
                        or (sibling.string.strip() == "et")
                        or (sibling.string.strip() == ",")
                    ):
                        logger.info(
                            "The sibling of the paragraph link is 'and'. Will add one more paragraph"
                        )
                        # The citation is citation to paragraph range
                        target_number = sibling.next_sibling.get_text(strip=True)
                        logger.debug(f"Adding target paragraph: {target_number}")
                        links_with_ranges.append(
                            (l, [paragraph.get_text(strip=True), target_number])
                        )
                        i += 2
                    elif sibling:
                        # Postulate that the citation is citation to paragraph and has a link to a single paragraph
                        logger.info(
                            "The sibling of the paragraph link is not 'to'. Will extract the single paragraph"
                        )
                        logger.debug(
                            f"Extracted target paragraph: {paragraph.get_text(strip=True)}"
                        )
                        links_with_ranges.append((l, [paragraph.get_text(strip=True)]))
                        i += 1
                else:  # the case_link is a case_link but we don't have a corresponding paragraph link
                    logger.info(
                        "The case link is a case link but we don't have a corresponding paragraph link."
                        "Will add it as a case link, and 'None' as the target paragraph"
                    )
                    links_with_ranges.append((l, None))
                    # counter i does not change at this point in order to check
                    # the next case_link with the same paragraph link
            except ValueError:
                logger.debug(
                    f"ValueError for case link: {unidecode(case_link.get_text(strip=True))}. "
                )
        logger.debug(f">>>Finished _extract_paragraph_ranges")
        for l in links_with_ranges:
            logger.debug(
                f"Link: {unidecode(l[0].get_text(strip=True))} - Target paragraphs: {l[1]}"
            )

        return links_with_ranges

    def _arrange_links(self, links: list) -> dict:
        """
        Arranges the links into a dictionary with the paragraph numbers as keys and the links as values
        :param links: list
            the list of links to arrange
        :return: dict
            a dictionary with the paragraph numbers as keys and the links as values, where each entry is of the form
            {paragraph_number: {'case_links': [list of case links], 'number_links': [list of paragraph number links]}}
        """

        # isolate those links whose text is not a number, ie, not paragraphs
        case_links = [l for l in links if not l.get_text(strip=True).isdigit()]
        # Check if the first case_link has the same case_number as the number of the case we are parsing
        # If it is the case, then delete it from the list
        # First convert casename using celex to caseid
        caseid = celex_to_caseid(self.case_name)
        # now is caseid equals to the first case_link, then remove it

        linked_case_text = case_links[0].get_text(strip=True)
        if re.search(r"\s", linked_case_text):
            linked_case_text = re.split(r"\s", linked_case_text)[0]
        linked_case_id = unidecode(linked_case_text)
        logger.info(f"Linked case is: {linked_case_id} ")
        # The first link to a case in the text can be a link to the case itself in the introduction,
        # where the relevant text just describes, how the case came up. We don't need such an incident,
        # so we just remove it
        if case_links and caseid == linked_case_id:
            case_links.pop(0)
        # Now isolate links that only contain numbers
        number_links = [l for l in links if l.get_text(strip=True).isdigit()]

        links_per_paragraph = {}

        # Initialize links_per_paragraph with a empty 'case_link' and 'paragraph_number' lists for each paragraph number
        # where the actual paragraph number is extracted for the case_links
        for c in case_links:
            try:
                paragraph_number = self._extract_paragraph_number(c)
                logger.debug(f"Extracted paragraph number: {paragraph_number}")
                # Initialize with empty reference dictionaries only if the resulting paragraph is non-null
                (
                    links_per_paragraph.update(
                        {paragraph_number: {"case_links": [], "number_links": []}}
                    )
                    if paragraph_number
                    else links_per_paragraph
                )
            except AttributeError:
                logger.error(
                    f"No paragraph number found for case link: {unidecode(c.get_text(strip=True))}. "
                    f"The case link might be in the conclusion as in 62019CJ0883"
                )
                continue

        logger.debug(f"Paragraph dictionary {links_per_paragraph}")
        # Iterate case links and add the paragraph number of the paragraph where the link is located, to the dictionary
        for c in case_links:
            try:
                paragraph_number = self._extract_paragraph_number(c)
                if paragraph_number:
                    links_per_paragraph[paragraph_number]["case_links"].append(c)
            except AttributeError:
                continue
            except KeyError:
                logger.error(f"KeyError for paragraph number: {paragraph_number}.")
                sys.stderr.write(
                    f"Exited with KeyError for paragraph number: {paragraph_number}.\n"
                )
                sys.exit(-1)

        for n in number_links:
            paragraph_number = self._extract_paragraph_number(n)
            # Add n to the list of number links at paragraph_number
            # if paragraph_number is not None and is in links_per_paragraph, otherwise continue
            (
                links_per_paragraph[paragraph_number]["number_links"].append(n)
                if paragraph_number is not None
                and paragraph_number in links_per_paragraph
                else None
            )

        return links_per_paragraph

    def _parse_citations(self):
        """
        Parses the HTML file and extracts the citations to other cases and to relevant paragraphs
        if those citations exist
        :return:
            citations: the citations to other cases and to relevant paragraphs if those citations exist and
        """
        citations = []
        try:
            with open(self.filepath, mode="r", encoding="utf-8") as f:
                contents = f.read()
                soup = bs(contents, "html.parser")
                # Find all <a> elements with class "coj-CourtLink" and where the text does not contain "EU:", which is
                # an indication that the citation has a case number eg. C-123/12
                links = soup.find_all(
                    "a", class_="coj-CourtLink", string=lambda x: x and "EU:" not in x
                )
                if len(links) == 0:
                    logger.info(
                        f"Cannot parse citations for {self.filepath}. HTML file does not contain links"
                    )
                    return citations  # Which are empty but the code shall return

                links_dict = self._arrange_links(links)

                for par_number in links_dict.keys():
                    # extract the case links and the number links
                    case_links = links_dict[par_number]["case_links"]
                    number_links = links_dict[par_number]["number_links"]

                    logger.debug(
                        "Extracted %d case links and %d number links (links to paragraph numbers)"
                        % (len(case_links), len(number_links))
                    )
                    # extract the citations to singleton paragraphs and paragraph ranges
                    citation_links = self._extract_paragraph_ranges(
                        case_links, number_links
                    )

                    # As a last step, iterate over the citation_links, extract the CELEX numbers
                    # and the source paragraph numbers and add them to the citations list
                    logger.debug(
                        "Will iterate over the citation links and extract the CELEX numbers "
                        "and the source paragraph numbers"
                    )

                    for link_entry, target_numbers in citation_links:
                        # convert each link_entry to CELEX
                        logger.info(
                            "Calling caseid_to_celex with %s"
                            % unidecode(link_entry.get_text(strip=True))
                        )

                        target_celex = caseid_to_celex(link_entry.get_text(strip=True))
                        logger.info(f"Extracted target CELEX: {target_celex}")

                        # Filter: Only include allowed document types
                        # HTML parser only extracts case citations, but we verify it's a CJ judgment
                        if not is_allowed_document_type(target_celex):
                            logger.debug(
                                "Skipping citation to non-allowed document type %s from HTML parser"
                                % target_celex
                            )
                            continue

                        # For CJ judgments from HTML, we assume paragraphs (HTML parser extracts paragraph citations)
                        # Pages would be detected differently, but HTML parser focuses on paragraph links
                        citations.append(
                            {
                                "target": target_celex,
                                "from": [par_number],
                                "to": target_numbers,
                                "cited_unit": "paragraphs",
                                "unit": "paragraphs",
                            }
                        )
        except FileNotFoundError:
            logger.error("Neither HTML file is available at %s" % self.filepath)

        except Exception as e:
            logger.exception(f"Error {e} occured exiting")
            sys.exit(-1)

        return citations

    def read_citations_and_citing_paragraphs(self):
        """
        Reads the citations to other cases and to relevant paragraphs if those citations exist
        :return: a list of citation dictionaries, where each dictionary has the entries 'from', 'to', 'target_celex'
            and 'citation_unit'
        Note: This method has ended up being an interface for the actual method that does the parsing
        """

        citations = self._parse_citations()
        logger.info(f"Extracted {len(citations)} citations from {self.filepath}")
        return citations


class EcjXmlParser(object):
    """
    Parses XML metadata for a specific file passed as initialization parameter and returns values of interest
    """

    XPATH_CELEX = "//WORK/SAMEAS//TYPE[text()='celex']/../IDENTIFIER/text()"  # OK
    XPATH_ECLI = "//WORK/SAMEAS//TYPE[text()='ecli']/../IDENTIFIER/text()"  # ook
    XPATH_FULL_TITLE = "/NOTICE/EXPRESSION/EXPRESSION_TITLE/VALUE/text()"  # //EXPRESSION_TITLE/LANG[text()='en']/../VALUE
    # XPATH_SHORT_TITLE = "/NOTICE/EXPRESSION/EXPRESSION_TITLE_ALTERNATIVE/VALUE/text()"
    XPATH_SHORT_TITLE = "//EXPRESSION/EXPRESSION_TITLE_ALTERNATIVE/VALUE/text()"  # cannot find in 62016CJ0641 exists in 61975CJ0111!
    XPATH_SHORT_TITLE_ALT = (
        "//EXPRESSION/EXPRESSION_USES_LANGUAGE/OP-CODE[text()='ENG']/"
        "../../EXPRESSION_TITLE_ALTERNATIVE/VALUE/text()"
    )
    XPATH_CREATED_BY = "//CREATED_BY/PREFLABEL/text()"  # the court - OK
    XPATH_CITATION = (
        "//WORK_CITES_WORK//URI/TYPE[text()='celex']/../IDENTIFIER/text()"  # ok
    )
    XPATH_CITING_PARAGRAPHS = "//WORK_CITES_WORK//URI/IDENTIFIER[text()='{celex}']/../../..//FRAGMENT_CITING_SOURCE/text()"
    # gives all the citing paragraphs (or pages for older docs) - 61983CJ0112
    # the celex in {} is the celex number of the *citation target*
    # ok
    XPATH_CITED_PARAGRAPHS = "//WORK_CITES_WORK//URI/IDENTIFIER[text()='{celex}']/../../..//FRAGMENT_CITED_TARGET/text()"
    # gives the cited paragraphs from the target document - - 61983CJ0112
    # ok
    XPATH_CITATION_ROOT = "//WORK_CITES_WORK"  # OK
    XPATH_CELEX_FROM_CITATION_ROOT = (
        ".//URI/TYPE[text()='celex']/../IDENTIFIER/text()"  # has to be ok
    )
    XPATH_LANGUAGE_OF_CASE = (
        "//RESOURCE_LEGAL_USES_ORIGINALLY_LANGUAGE/OP-CODE/text()"  # OK
    )
    XPATH_DOCUMENT_DATE = (
        "//WORK_DATE_DOCUMENT/VALUE/text()"  # take only the first value - OK
    )
    XPATH_APPLICATION_DATE = "//RESOURCE_LEGAL_DATE_REQUEST_OPINION/VALUE/text()"  # OK
    XPATH_SUBJECT_MATTER = "//WORK_HAS_EXPRESSION[1]/EMBEDDED_NOTICE[1]//RESOURCE_LEGAL_IS_ABOUT_SUBJECT-MATTER//PREFLABEL/text()"  # OK
    # The following returns a hierarchy of up to three-four nested concepts
    # The concepts are not really nested in this XML but are denoted by CASE-LAW_IS_ABOUT_CONCEPT_CASE-LAW_1, CASE-LAW_IS_ABOUT_CONCEPT_CASE-LAW_2
    # CASE-LAW_IS_ABOUT_CONCEPT_CASE-LAW_3
    XPATH_CASE_LAW_ABOUT = "//WORK_HAS_EXPRESSION[1]/EMBEDDED_NOTICE[1]//CASE-LAW_IS_ABOUT_CONCEPT_CASE-LAW"
    XPATH_CASE_LAW_ABOUT_NEW = "//WORK_HAS_EXPRESSION[1]/EMBEDDED_NOTICE[1]//CASE-LAW_IS_ABOUT_CONCEPT_NEW_CASE-LAW"
    # the last two do not exist and we have //CASE-LAW_IS_ABOUT_CONCEPT//PREFLABEL, eg. for 62016CJ0642
    XPATH_ADVOCATE_GENERAL = "//CASE-LAW_DELIVERED_BY_ADVOCATE-GENERAL/SAMEAS//IDENTIFIER/text()"  # OK but multivalues are concatenated with ';'
    XPATH_RAPPORTEUR = "//CASE-LAW_DELIVERED_BY_JUDGE/SAMEAS//IDENTIFIER/text()"  # OK
    XPATH_COURT = "//WORK_CREATED_BY_AGENT/PREFLABEL/text()"  # Otherwise: Author - OK, Court of Justice
    XPATH_DOCUMENT_TYPE = "//WORK/WORK_HAS_RESOURCE-TYPE//PREFLABEL[1]"
    XPATH_DOCUMENT_TYPE_NEWER = (
        "//WORK/RESOURCE_LEGAL_HAS_TYPE_ACT_CONCEPT_TYPE_ACT/PREFLABEL[1]"
    )

    # "//RESOURCE_LEGAL_HAS_TYPE_ACT_CONCEPT_TYPE_ACT/PREFLABEL/text()"
    XPATH_PROCEDURE_TYPE = (
        "//CASE-LAW_HAS_TYPE_PROCEDURE_CONCEPT_TYPE_PROCEDURE/PREFLABEL/text()"  # OK
    )
    XPATH_APPLICANT = "//CASE-LAW_REQUESTED_BY_AGENT/PREFLABEL/text()"  # Missing?
    XPATH_APPLICANT_ID = "//CASE-LAW_REQUESTED_BY_AGENT/IDENTIFIER/text()"  # Missing?
    XPATH_DEFENDANT = (
        "//CASE-LAW_DEFENDED_BY_AGENT/PREFLABEL/text()"  # exists in some old data
    )
    XPATH_DEFENDANT_ID = "//CASE-LAW_DEFENDED_BY_AGENT/IDENTIFIER/text()"  # old data
    XPATH_OBSERVATIONS = "//CASE-LAW_COMMENTED_BY_AGENT/PREFLABEL/text()"  # OK
    XPATH_OBSERVATIONS_ID = "//CASE-LAW_COMMENTED_BY_AGENT/IDENTIFIER/text()"  # OK
    XPATH_NATIONAL_COURT = "//CASE-LAW_NATIONAL-JUDGEMENT/VALUE/text()"  # OK
    XPATH_KEYWORDS = "//MANIFESTATION_CASE-LAW_KEYWORDS/VALUE/text()"
    # This is not present and could probably be extracted from the title
    # it is the fourth part when splitting by #

    # CITATION_FILTER = r'[136]\d{4}[CTFLE]' # more restrictive version
    CITATION_FILTER = r"[136]\d{4}"
    NATIONAL_COURT = r"(\*\w\d\*)"

    IDS_TO_IGNORE = ["EM", "IC"]  # Member States  # Institutions

    def __init__(self, filepath):
        try:
            self._tree = etree.parse(filepath)
            self._filepath = filepath
            self._keywords = []
            self._celex = None
        except lxml.etree.XMLSyntaxError:
            logger.error(
                "XML metadata parsing failed for %s. Continuing with next file."
                % filepath
            )
            with open("nometadata.txt", mode="a+") as nometa:
                # trying to write the CELEX of the failed
                nometa.write(filepath.split(os.pathsep)[-1] + "\n")
                raise NoMetadataError

    def __del__(self):
        self._tree = None

    def xpath(self, xpathstr):
        """
        Run Xpath on the metadata
        :param xpathstr: specifies the XPath string
        :return: node or list(node)
        """

        return self._tree.xpath(xpathstr)

    def read_celex(self):
        celex_node = self.xpath(EcjXmlParser.XPATH_CELEX)
        if len(celex_node) > 0:
            self._celex = celex_node[0]
            return celex_node[0]

    def read_ecli(self):
        ecli_node = self.xpath(EcjXmlParser.XPATH_ECLI)
        if len(ecli_node) > 0:
            return ecli_node[0]

    def read_caseno(self, celex):
        return celex_to_caseid(celex)

    def read_full_title(self):
        """
        Returns the full title of the document. The full title contains other data, such as keywords, procedure type,
        date etc. Usually the fields are separated by the '#' symbol and full title is the second value
         :return The full title
         :rtype str
        """
        full_title = self.xpath(EcjXmlParser.XPATH_FULL_TITLE)
        full_title_final = None
        try:
            title_pieces = full_title[0].split("#")
            title_pieces = [t.replace("\xa0", " ") for t in title_pieces]
            full_title_final = title_pieces[1].strip()
            if len(title_pieces) == 5:
                self._keywords = [
                    k.strip(" .")
                    for k in re.split(r"\s[\-\u2013\u2014]\s", title_pieces[3])
                ]
            elif len(title_pieces) == 4:
                self._keywords = [
                    k.strip(" .")
                    for k in re.split(r"\s[\-\u2013\u2014]\s", title_pieces[2])
                ]
        except IndexError:
            logger.error("Case doesn't have a full title")
        finally:
            return full_title_final

    def read_short_title(self):
        short_title_node = self.xpath(EcjXmlParser.XPATH_SHORT_TITLE)

        if len(short_title_node) == 1:
            return short_title_node[0]
        elif len(short_title_node) > 1:
            # locate the english short title
            short_title_node = self.xpath(EcjXmlParser.XPATH_SHORT_TITLE_ALT)
            if len(short_title_node) > 0:
                return short_title_node[0]
            else:
                # TODO: perhaps isolate the parties' names before the v. We could leave them blank as the court uses
                #  structured references
                logger.info("Case doesn't have a short title")
                return ""
        elif len(short_title_node) == 0:
            logger.info("Case doesn't have a short title")
            return ""

    def read_keywords(self):
        # We might have extracted keywords from the full title
        if len(self._keywords) > 0:
            return self._keywords
        else:
            # We deliberately avoid the XML section marked as keywords as this seems to be a bit unrelated
            # according to Urska
            return None

    def read_citations(self):
        """
        Extracts the citations from the metadata. Returns citations passing through the CITATION_FILTER, which
        is a regex that matches the CELEX of the cited case, allowing both judgments, treaties and articles.

        Returns
        -------
        list(str) of CELEX numbers corresponding to citation roots
        """
        # How are citation_roots extracted?
        # 1. Find all citation roots, which are the nodes r
        # 2. Filter out the ones that don't have a CELEX like string in the CELEX_FROM_CITATION_ROOT
        # 3. Return the list of citation roots

        citation_roots = [
            r
            for r in self.xpath(EcjXmlParser.XPATH_CITATION_ROOT)
            if
            # Included in the list comprehension if it matches the filter
            re.search(
                re.compile(EcjXmlParser.CITATION_FILTER),
                # The expression below gets the CELEX of the cited case
                (
                    r.xpath(EcjXmlParser.XPATH_CELEX_FROM_CITATION_ROOT)[0]
                    if len(r.xpath(EcjXmlParser.XPATH_CELEX_FROM_CITATION_ROOT)) > 0
                    # ... or an empty string if there is no CELEX
                    else ""
                ),
            )
        ]

        return citation_roots

    def read_citations_and_citing_paragraphs(self, include_all=True):
        """
        Reads the citations from the XML and returns a list of dictionaries for each citation,
        where each dictionary contains the target celex, a list of paragraphs citing the target and
        if exists, the target paragraphs that is referenced
        :param include_all: bool, default True
            If True, the method will process citations to all document types (treaties, regulations,
            directives, legal acts, and case law). If False, only case law citations are included.
        :return:
        list of citation dictionaries
        """
        citations_list = []
        roots = self.read_citations()
        # Log the number of citations
        logger.debug("Number of citations:%d" % len(roots))

        for root in roots:
            celex = root.xpath(EcjXmlParser.XPATH_CELEX_FROM_CITATION_ROOT)[0]

            # Filter to only include allowed document types: CJ judgments, Directives, Regulations, Delegated Regulations
            if not is_allowed_document_type(celex):
                logger.info("Skipping citation to non-allowed document type %s" % celex)
                continue

            citation_pairs = root.xpath("./ANNOTATION")
            if len(citation_pairs) != 0:
                citations_list = self._citation_and_citing_paragraphs(
                    citations_list, celex, citation_pairs
                )
            else:
                logger.warning("There are no target references to paragraphs")
                logger.debug("Will proceed with processing the HTML file")

        return citations_list

    def _detect_unit(self, citation_string: str) -> str:
        """
        Detects the unit type from a citation string. Handles both prefix formats (A107, N107)
        and nested formats (107P1, 03P5L2) where numbers are followed by unit letters.
        Common prefixes: A=articles, N=paragraphs, P=pages, R=recitals, C=chapters, T=titles, L=lines

        :param citation_string: The citation string to analyze
        :return: The detected unit type (articles, paragraphs, pages, recitals, chapters, titles, lines, or unknown)
        """
        if not citation_string:
            return "unknown"

        citation_string = citation_string.strip().upper()

        # Check for explicit unit prefixes at the start (A107, N107, P5, etc.)
        if citation_string.startswith("A"):
            return "articles"
        elif citation_string.startswith("N"):
            return "paragraphs"
        elif citation_string.startswith("P"):
            return "pages"
        elif citation_string.startswith("R"):
            return "recitals"
        elif citation_string.startswith("C"):
            return "chapters"
        elif citation_string.startswith("T"):
            return "titles"
        elif citation_string.startswith("L"):
            return "lines"

        # Handle nested formats like "107P1" (Article 107, Paragraph 1) or "03P5L2" (Article 03, Paragraph 5, Line 2)
        # Pattern: number followed by unit letter(s)
        # For "107P1", the first unit is implied to be Article (A) if it starts with digits
        # For "03P5L2", same logic applies

        # Check if it starts with digits (nested format without explicit prefix)
        if re.match(r"^\d+", citation_string):
            # Look for the first unit letter after a number
            # Pattern: digits followed by unit letter (P, N, L, R, C, T)
            match = re.search(r"\d+([PNLRCT])", citation_string)
            if match:
                unit_letter = match.group(1)
                unit_map = {
                    "P": "paragraphs",  # In nested format, P after number usually means paragraph
                    "N": "paragraphs",
                    "L": "lines",
                    "R": "recitals",
                    "C": "chapters",
                    "T": "titles",
                }
                # However, if the string starts with digits and has P, it might be Article X Paragraph Y
                # Check if there's a pattern like "107P1" - this is likely Article 107, Paragraph 1
                if re.match(r"^\d+P\d+", citation_string):
                    # This is likely Article X, Paragraph Y format
                    return "articles"
                return unit_map.get(unit_letter, "paragraphs")
            # If no unit letter found but starts with digits, likely an article number
            return "articles"

        # Default to paragraphs for backward compatibility
        return "paragraphs"

    def _citation_and_citing_paragraphs(self, citations_list, celex, citation_pairs):
        """
        Replicates the code in lines 610/657 in a function
        :param celex: the celex number of the cited case
        :param citation_pairs: the citation pairs
        :return: list of citation dictionaries
        """

        for citation in citation_pairs:
            cited = citation.xpath(".//FRAGMENT_CITED_TARGET/text()")
            citing = citation.xpath(".//FRAGMENT_CITING_SOURCE/text()")

            logger.debug("pararagraph or paragraphs %s cited: %s" % (citing, cited))

            if len(cited) > 0:
                if len(citing) > 0:
                    cited_par = cited[0]
                    citations_dict = {}
                    cited_unit = self._detect_unit(cited_par)
                    citation_string = citing[0]

                    citations_dict["from"] = self._citing_paragraphs(citation_string)
                    citations_dict["to"] = self._citing_paragraphs(cited_par)
                    citations_dict["target"] = celex
                    citations_dict["cited_unit"] = cited_unit
                    citations_dict["unit"] = self._detect_unit(citation_string)
                    citations_list.append(citations_dict)
                else:
                    cited_par = cited[0]
                    citations_dict = {}
                    cited_unit = self._detect_unit(cited_par)

                    citations_dict["to"] = self._citing_paragraphs(cited_par)
                    citations_dict["target"] = celex
                    citations_dict["cited_unit"] = cited_unit
                    citations_list.append(citations_dict)

            elif len(citing) > 0:
                citations_dict = {}
                citation_string = citing[0]
                detected_unit = self._detect_unit(citation_string)

                output_pars = self._citing_paragraphs(citation_string)
                citations_dict.setdefault("from", output_pars)
                citations_dict.setdefault("target", celex)
                citations_dict["unit"] = detected_unit
                citations_list.append(citations_dict)
        return citations_list

    def _citing_paragraphs(self, citation_string):
        citing_pars = self._preconditionListText(citation_string)
        output_pars = []
        for par in citing_pars:
            logger.debug("Adding citing paragraph or range %s" % par)
            # Append to the output the result of self._processRange(par) if not None else continue
            par_range = self._processRange(par)
            output_pars += par_range if par_range is not None else []
            # output_pars += self._processRange(par)
        return output_pars

    def _preconditionListText(self, text):
        """
        Preconditions the list of citation units (paragraphs/pages/articles/etc.) by removing
        the unit prefix and splitting the sentence. Supports articles (A), paragraphs (N),
        pages (P), recitals (R), chapters (C), titles (T), and lines (L).
        Also handles nested formats like "107P1" (Article 107, Paragraph 1) and "03P5L2" (Article 03, Paragraph 5, Line 2).
        For nested formats, extracts only the base number (e.g., "08" from "08P2").
        """
        # convert 3 - 5 to 3-5 so that it will be unaffected when we split by space
        pageListText = re.sub(r"(\s+)?-(\s+)?", "-", text)

        # Check if this is a nested format (starts with digits, e.g., "107P1", "03P5L2", "08P2")
        # For nested formats, extract only the base number (remove subunit parts like P2, L2, etc.)
        if re.match(r"^\d+[PNLRCT]", pageListText):
            # Extract only the first number part (e.g., "8" from "08P2")
            match = re.match(r"^(\d+)", pageListText)
            if match:
                base_number = str(
                    int(match.group(1))
                )  # Convert "08" to "8", remove leading zeros
                pageListText = re.sub(r"^\d+[PNLRCT].*", base_number, pageListText)
            pageListText = re.sub(r"[J:]", "", pageListText)
            return pageListText.split()

        # For prefix formats (A107, N107, P5, etc.), remove the starting unit prefix
        # The pattern matches one or more unit prefixes (A, P, N, R, C, T, L) optionally followed by whitespace
        pageListText = re.sub(r"^[APNRC,]+T?\s*", "", pageListText)
        # Also handle L (lines) prefix
        pageListText = re.sub(r"^L\s*", "", pageListText)
        # WTF some par numbers had a J or :
        pageListText = re.sub(r"[J:]", "", pageListText)
        return pageListText.split()

    def _processRange(self, listEntry):
        """
        Processes a single list entry and returns a range if the entry is separated by hyphen.
        Handles ranges for articles, paragraphs, pages, recitals, chapters, titles, and lines.
        Also handles nested formats like "107P1-107P3" (Article 107, Paragraphs 1-3).
        For nested formats, extracts only the base number (e.g., "08" from "08P2").
        :param listEntry:
        :return: a list of string containing the source unit numbers
        """
        # Check if this is a nested format (contains unit letters like P, L, etc.)
        if re.search(r"[PNLRCT]\d+", listEntry):
            # Extract only the base number from nested formats
            # For "08P2", extract "8"; for "107P1-107P3", extract "107" from both sides
            if "-" in listEntry:
                # Handle ranges like "08P2-08P5" or "107P1-107P3"
                parts = listEntry.split("-")
                base_numbers = []
                for part in parts:
                    match = re.match(r"^(\d+)", part)
                    if match:
                        # Remove leading zeros: "08" -> "8"
                        base_numbers.append(str(int(match.group(1))))
                if len(base_numbers) == 2:
                    # If we have a range, try to expand it
                    try:
                        fromNum = int(base_numbers[0])
                        toNum = int(base_numbers[1])
                        r = list(range(fromNum, toNum + 1))
                        return [str(num) for num in r]
                    except ValueError:
                        return base_numbers
                return base_numbers
            else:
                # Single nested format like "08P2" - extract just "8"
                match = re.match(r"^(\d+)", listEntry)
                if match:
                    # Remove leading zeros: "08" -> "8"
                    return [str(int(match.group(1)))]
                return [listEntry]

        # split to find the ends of the range
        listEntry_split = listEntry.split("-")
        if self._celex.startswith(("3", "6")) and len(listEntry_split) > 1:
            try:
                # Remove unit prefixes (A=articles, P=pages, N=paragraphs, R=recitals, C=chapters, T=titles, L=lines)
                # and spaces from the beginning and end of the string
                fromNum = int(listEntry_split[0].strip("APNRC L "))
                toNum = int(listEntry_split[1].strip("APNRC L "))
                r = list(range(fromNum, toNum + 1))
                return [str(num) for num in r]
            except ValueError:
                return
        elif self._celex.startswith("1"):
            return None
        else:
            return listEntry_split  # no range just a single number

    def _remove_unwanted_entries(self, d):
        """Removes some of the entries in e.g. national submissions, because those are superfluous. Like for example
        'Member States'.
        :param d, dictionary of entities, in form {'GR': 'Greece'}
        """
        for id in self.IDS_TO_IGNORE:
            if id in d:
                del d[id]

    def read_language_of_case(self):
        return self.xpath(EcjXmlParser.XPATH_LANGUAGE_OF_CASE)

    def read_document_date(self):
        return self.xpath(EcjXmlParser.XPATH_DOCUMENT_DATE)

    def read_application_date(self):
        return self.xpath(EcjXmlParser.XPATH_APPLICATION_DATE)

    def read_subject_matter(self):
        return self.xpath(EcjXmlParser.XPATH_SUBJECT_MATTER)

    def read_case_law_is_about(self):
        concepts = self.xpath(EcjXmlParser.XPATH_CASE_LAW_ABOUT)
        # concepts_dict is a dictionary of the form
        # directory_code: level description, e.g.
        # "3.02.01": "Legal proceedings / Actions for annulment / Acts which may be the subject of an action for annulment"
        # The directory code is always the code of the last level (most specific one).
        concepts_dict = OrderedDict()

        # We try to read the XML metadata documents and flatten the hierarchy. We actually run through all possible
        # directory levels read the level description, append the description in a list and at the end we concat
        # list items with " / "
        # We are also reading the level identifier and actually keep the innermost identifier which becomes a key to the dictionary
        for c in concepts:
            logger.debug("Concept %s" % str(c))
            level_1 = c.xpath("CASE-LAW_IS_ABOUT_CONCEPT_CASE-LAW_1/PREFLABEL/text()")[
                0
            ]
            # here is the level identifier
            case_law_directory_code = c.xpath(
                "CASE-LAW_IS_ABOUT_CONCEPT_CASE-LAW_1/IDENTIFIER/text()"
            )[0]
            sublist = [level_1]

            for l in range(2, 6):
                try:
                    level = c.xpath(
                        "CASE-LAW_IS_ABOUT_CONCEPT_CASE-LAW_%d/PREFLABEL/text()" % l
                    )[0]
                    # We overwrite the identifier to keep the mos specific one
                    case_law_directory_code = c.xpath(
                        "CASE-LAW_IS_ABOUT_CONCEPT_CASE-LAW_%d/IDENTIFIER/text()" % l
                    )[0]
                    sublist.append(level)
                except IndexError:
                    # IndexError will mean that the list has zero length and there are no further levels
                    break

            # join level labels to obtain something similar to what Eur-Lex has
            concepts_dict[case_law_directory_code] = " / ".join(sublist)

        # Repeat the procedure above for the directory after 2003
        if len(concepts) == 0:
            concepts = self.xpath(EcjXmlParser.XPATH_CASE_LAW_ABOUT_NEW)

            for c in concepts:
                logger.debug("Concept %s" % str(c))
                level_1 = c.xpath(
                    "CASE-LAW_IS_ABOUT_CONCEPT_NEW_CASE-LAW_1/PREFLABEL/text()"
                )[0]
                case_law_directory_code = c.xpath(
                    "CASE-LAW_IS_ABOUT_CONCEPT_NEW_CASE-LAW_1/IDENTIFIER/text()"
                )[0]
                sublist = [level_1]

                for l in range(2, 7):
                    try:
                        level = c.xpath(
                            "CASE-LAW_IS_ABOUT_CONCEPT_NEW_CASE-LAW_%d/PREFLABEL/text()"
                            % l
                        )[0]
                        case_law_directory_code = c.xpath(
                            "CASE-LAW_IS_ABOUT_CONCEPT_NEW_CASE-LAW_%d/IDENTIFIER/text()"
                            % l
                        )[0]
                        sublist.append(level)
                    except IndexError:
                        # IndexError will mean that the list has zero length and there is no 2nd level
                        continue
                concepts_dict[case_law_directory_code] = " / ".join(sublist)
        return concepts_dict

    def read_advocate_general(self):
        return self.xpath(EcjXmlParser.XPATH_ADVOCATE_GENERAL)

    def read_rapporteur(self):
        return self.xpath(EcjXmlParser.XPATH_RAPPORTEUR)

    def read_court(self):
        return (
            self.xpath(EcjXmlParser.XPATH_COURT)[0]
            if len(self.xpath(EcjXmlParser.XPATH_COURT)) > 0
            else "Court of Justice"
        )

    def read_document_type(self):
        # Get the document type, e.g. Judgment, but the Xpath changes for newer documents
        try:
            element = self.xpath(EcjXmlParser.XPATH_DOCUMENT_TYPE)[0]
        except IndexError:
            logger.debug("Will try to use the alternative Xpath!")
            element = self.xpath(EcjXmlParser.XPATH_DOCUMENT_TYPE_NEWER)[0]

        # Need the element's text but have to make sure that is makes sense
        return element.text if isinstance(element, lxml.etree._Element) else element

    def read_procedure_type(self):
        """
        Reads the procedure type from the XML metadata. The procedure type is a list of dictionaries, where each
        dictionary has the keys 'id', 'type', 'result' and 'result_id'. The 'id' is the identifier of the procedure like
        'REFER_PREL_URG', 'type' is the type of the procedure like 'Preliminary reference',
        'result' is the result of the procedure like 'decision unnecessary' and 'result_id' is
        the identifier of the result

        Returns
        -------
        list of dictionaries {'id': str, 'type': str, 'result': str, 'result_id': str}
        """
        procedure_types = self.xpath(EcjXmlParser.XPATH_PROCEDURE_TYPE)

        # Note: When the procedure has a valid result then the XML has identifier of the form "ANNU=NL" where
        # whatever follows the "=" sign is the identifier for the result!
        # We don't want this right now so we throw away the result identifier
        procedure_type_ids = self.xpath(
            EcjXmlParser.XPATH_PROCEDURE_TYPE.replace("PREFLABEL", "IDENTIFIER")
        )

        procedure_list = []
        for type_id, type in zip(procedure_type_ids, procedure_types):
            logger.debug("Procedure type id: %s - type:%s" % (type_id, type))
            pieces = re.split(r"\s[-–]\s", type)
            logger.debug("Pieces: %s" % pieces)
            procedure_dict = {
                "id": type_id,
                "type": None,
                "result": None,
                "result_id": None,
            }
            if (len(pieces) > 1 and (pieces[1] != "urgent procedure")) or (
                len(pieces) > 1 and (pieces[1] != "Procédure d'urgence")
            ):

                if len(pieces) == 3:
                    # The following appeared in 62016CJ0452, where the Court gives two procedure types
                    # Preliminary reference - urgent procedure
                    # Preliminary reference - urgent procedure - decision unnecessary
                    # for the last one we just 'keep decision unnecessary', which basically is in accordance to
                    # <IDENTIFIER>REFER_PREL_URG=NL</IDENTIFIER>
                    procedure = pieces[0]
                    result = pieces[2]
                else:
                    (procedure, result) = pieces

                type_pieces = type_id.split("=")
                procedure_dict["id"] = type_pieces[0]
                procedure_dict["type"] = procedure
                procedure_dict["result"] = result
                procedure_dict["result_id"] = (
                    type_pieces[1] if len(type_pieces) == 2 else None
                )
            else:
                procedure = pieces[0].strip()
                procedure_dict["type"] = procedure
            procedure_list.append(procedure_dict)

            # convert the dict items to frozen sets
            set_of_dicts = set(frozenset(d.items()) for d in procedure_list)
            # convert the frozen sets back to dicts
            procedure_list = [dict(s) for s in set_of_dicts]
        return procedure_list

    def read_applicant(self):
        applicant_labels = self.xpath(EcjXmlParser.XPATH_APPLICANT)
        applicant_ids = self.xpath(EcjXmlParser.XPATH_APPLICANT_ID)
        applicants = dict(list(zip(applicant_ids, applicant_labels)))
        self._remove_unwanted_entries(applicants)
        return applicants

    def read_defendant(self):
        defendant_labels = self.xpath(EcjXmlParser.XPATH_DEFENDANT)
        defendant_ids = self.xpath(EcjXmlParser.XPATH_DEFENDANT_ID)
        defendants = dict(list(zip(defendant_ids, defendant_labels)))
        self._remove_unwanted_entries(defendants)
        return defendants

    def read_observations(self):
        observation_labels = self.xpath(EcjXmlParser.XPATH_OBSERVATIONS)
        observation_ids = self.xpath(EcjXmlParser.XPATH_OBSERVATIONS_ID)

        # The observations XPath, returns a list of values. Sometimes the values that it returns are of the form:
        # Member States, Commission, Greece, Institutions (CELEX:62003CJ0329)
        # In the above list, both "Member States" and "Institutions" are superfluous and for the purposes of our
        # database we will only keep the actual member state and the name of the institution

        observations = dict(list(zip(observation_ids, observation_labels)))
        self._remove_unwanted_entries(observations)
        return observations

    def read_national_court(self):
        court = self.xpath(EcjXmlParser.XPATH_NATIONAL_COURT)
        if len(court) == 0:
            return None
        # the following works for newer Python versions otherwise six.moves.html_parser shall be used
        html_text = html.unescape(court)[0]
        soup = bs(html_text, "lxml")
        # Now it becomes hacky :)
        # Usually they indicate the national court with *A9* or *P1* but can have other indications
        # We clean this part and get the court

        results = soup.find_all(string=re.compile(self.NATIONAL_COURT))
        # Process mult

    def is_advocate_general_opinion(self):
        return True if self._celex and "CC" in self._celex else False


if __name__ == "__main__":
    parser = Reader(root="judgments")
    parser.process(output_dir="results")
    # To merge all batch result files into one:
    # parser.merge_result_batches("results", "all_results.json")
    # print(json.dumps(parser.values_from_xml(parser.xmlfiles[-5]), indent=4))
