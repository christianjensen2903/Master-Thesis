import logging
import sys

from bs4 import BeautifulSoup as bs
from bs4 import NavigableString
import re
from lxml import etree  # type: ignore
import os, os.path


def celex_from_path(path):
    casedir = os.path.dirname(path)
    # The following works as long as the paths to judgment.html end up in "<celex>/EN/judgment.html"
    celex = os.path.split(os.path.split(casedir)[0])[1]

    return celex


class NoGroundsError(Exception):
    pass


class NoNumbersError(Exception):
    pass


class ParagraphNumberingError(Exception):
    pass


class ECJProcessor(object):
    _soup = None
    _version_dict = {"90-00s": 1, "00-03s": 2, "05-now": 3}
    _version = None

    NUM_PATT = r"^\s*(?P<number>{parno})\.\s+"
    NUM_PATT_NO_DOT = r"^\s*(?P<number>{parno})(\s+|[^\W\d_])"

    def __init__(self, path):
        self._path = path
        self._xml_path = self.find_xml()

        if self._xml_path:
            with open(self._xml_path, mode="br") as fp:
                utf8_parser = etree.XMLParser(encoding="utf-8")
                self._tree = etree.fromstring(fp.read(), parser=utf8_parser)
        else:
            self._tree = None

        self._grounds_tree = None
        if os.path.exists(self._path):
            with open(self._path, encoding="utf-8") as file:
                self._soup = bs(file.read(), "lxml")
        else:
            self._soup = None
        self._total_paragraphs = 0
        self._version = self.detect_version()

    @property
    def soup(self):
        return self._soup

    @property
    def version(self):
        return self._version

    @version.setter
    def version(self, v):
        self._version = v

    @property
    def total_paragraphs(self):
        if self._total_paragraphs == 0:
            self._total_paragraphs = self._detect_total_paragraphs(self.NUM_PATT)
        return self._total_paragraphs

    def find_xml(self) -> str | None:
        metadata_path = self._path.replace("_judgment.html", "_metadata.xml")
        if os.path.exists(metadata_path):
            return metadata_path
        return None

    def select_patt_type(self, first_par):
        """Selects pattern type given the text of the first numbered paragraph as input.
        This can be in fact any numbered paragraph"""
        if re.search(
            ECJProcessor.NUM_PATT.format(parno=1), first_par, flags=re.U | re.MULTILINE
        ):
            return ECJProcessor.NUM_PATT
        elif re.search(
            ECJProcessor.NUM_PATT_NO_DOT.format(parno=1),
            first_par,
            flags=re.U | re.MULTILINE,
        ):
            return ECJProcessor.NUM_PATT_NO_DOT
        elif first_par.find(attrs={"class": "coj-count"}):
            return ECJProcessor.NUM_PATT_NO_DOT
        else:
            raise NoNumbersError

    def extract_grounds(self):
        if self._grounds_tree is None:
            grounds_element = self._tree.xpath(
                "//MANIFESTATION_CASE-LAW_GROUNDS/VALUE/text()"
            )
            if len(grounds_element) > 0:
                grounds_text = grounds_element[0].strip()
                self._grounds_tree = etree.fromstring(grounds_text)
                return self._grounds_tree
            else:
                return None
        return self._grounds_tree

    def extract_paragraphs(self, judgment_paragraphs):
        """
        Extracts <p> parts of the HTML, belonging to judgment_paragraphs passed as input
        We are trying to append to the previous numbered paragraph as possible
        NOTE: The paragraphs are based on the CSS class "C01PointnumeroteAltN", which appears after 2010
        """
        cur = judgment_paragraphs[0]
        pars = []
        if "count" in cur.attrs:
            # the judgment has a very new format with each paragraph in a table
            # follow a different procedure
            pars_temp = self._soup.find_all(
                "p", attrs={"id": re.compile(r"point\d")}
            )  # find p elements like <p class="count" id="point1">1</p>

            for p in pars_temp:
                row_parent = p.fetchParents("tr")[0]
                p_text = row_parent.find("p", attrs={"class": "normal"})
                pars.append(p_text)

        while cur.find_next_sibling("p", attrs={"class": "C01PointnumeroteAltN"}):
            try:
                par_text = " ".join(list(cur.stripped_strings))
                if len(par_text) > 0:
                    pars.append(par_text)
                cur = cur.find_next_sibling(
                    "p", attrs={"class": "C01PointnumeroteAltN"}
                )
            except AttributeError:
                break
        # add last element too
        par_text = " ".join(list(cur.stripped_strings))
        pars.append(par_text)
        # Now simply enumerate the paragraph list to produce a dictionary
        par_dict = {par_no: text for par_no, text in enumerate(pars, start=1)}
        return par_dict

    def read_paragraphs(self):
        """
        Read the paragraphs of the case passed as input

        Extraction was done by using the fulltext of the judgment passed together
        with XML metadata. This has stopped by the Court and we have to parse the HTML

        Returns
        -------
        paragraphs: dict
            A dictionary storing paragraph contents, indexed by paragraph number
        """
        grounds_tree = self.extract_grounds()

        # grounds_tree
        if grounds_tree is None:
            # raise NoGroundsError
            # self._detect_total_paragraphs(self.NUM_PATT) - Make no much sense to use

            # load the judgment as soup
            # isolate the actual judgment
            judgment_paragraphs = self._isolate_judgment()
            first_text = judgment_paragraphs[0].text
            num_pattern = self.select_patt_type(first_text)
            # isolate paragraphs
            # Perhaps try to iterate through judgment_paragraphs and concat consecutive paragraphs
            total_paragraphs, last_par_text = self._detect_total_paragraphs(
                num_pattern.format(parno=r"\d+")
            )
            # paragraphs = {0:None}
            # paragraph_range = iter(list(range(1, total_paragraphs)))

            # Try to navigate with BeautifulSoup
            paragraphs = self.extract_paragraphs(judgment_paragraphs)
        else:
            p_elements = grounds_tree.getchildren()
            i = 0
            par = p_elements[i]
            while not par.text.strip().startswith("1"):
                i += 1
                if i < len(p_elements):
                    par = p_elements[i]
                else:
                    raise NoNumbersError

            first_text = par.text.strip()
            num_pattern = self.select_patt_type(first_text)
            total_paragraphs, last_par_text = self._detect_total_paragraphs(
                num_pattern.format(parno=r"\d+")
            )

            paragraphs = {}
            paragraph_range = iter(list(range(1, total_paragraphs)))

            for i_par in paragraph_range:
                par_text, skipped = self.paragraph_extractor(i_par, num_pattern)
                paragraphs[i_par] = par_text
                if skipped:
                    next(paragraph_range)

            paragraphs[total_paragraphs] = last_par_text

        return paragraphs

    def detect_version(self):
        """
        NOTE: probably deprecated
        :return:
        """
        path = self._path

        try:
            with open(path, encoding="utf8") as html:
                soup = bs(html.read(), "lxml")
                div = soup.find("div", attrs={"id": "banner"})
        except UnicodeDecodeError:
            with open(path, encoding="latin-1") as html:
                soup = bs(html.read(), "lxml")
                div = soup.find("div", attrs={"id": "banner"})

        if div:
            self._version = self._version_dict["90-00s"]
        elif soup.find("p", attrs={"class": "C01PointnumeroteAltN"}):
            self._version = self._version_dict["05-now"]
        elif soup.find("table", attrs={"width": "100%"}):
            self._version = self._version_dict[
                "05-now"
            ]  # this is a first variant of that version
        elif soup.find("dt"):
            self._version = self._version_dict["00-03s"]
        # casefile.close()
        return self._version

    def paragraph_extractor(self, par_num, num_pattern):
        """
        Extract the paragraph text for paragraph with number `par_num`
        """
        skipped = False
        first_par = ECJProcessor.tree_xpath_regex(
            self._grounds_tree, num_pattern.format(parno=par_num)
        )[0]
        # TODO: The above should not really work with the HTMLs so perhaps change in BeautifulSoup
        try:
            next_par = ECJProcessor.tree_xpath_regex(
                self._grounds_tree,
                num_pattern.format(parno=par_num),
                following=True,
                following_no=num_pattern.format(parno=par_num + 1),
            )[0]
        except IndexError as e:
            # if the regex cannot find a paragraph starting with par_num+1, tree_path_regex will return an empty list
            # and thus the above expression will result in an IndexError. We repeat the procedure with par_num+2
            skipped = True
            try:
                next_par = ECJProcessor.tree_xpath_regex(
                    self._grounds_tree,
                    num_pattern.format(parno=par_num),
                    following=True,
                    following_no=num_pattern.format(parno=par_num + 2),
                )[0]
            except IndexError as e:
                raise ParagraphNumberingError(
                    "Paragraph numbering problem near paragraph {parno}".format(
                        parno=par_num
                    )
                )

        texts = []

        while first_par != next_par:
            # Skip possibly empty paragraphs
            if not hasattr(first_par, "text"):
                # Fallback
                first_par = ECJProcessor.tree_xpath_regex(
                    self._grounds_tree, num_pattern.format(parno=par_num)
                )[0]
            if first_par.text is None:
                first_par = first_par.getnext()
                continue
            texts.append(first_par.text.strip())
            first_par = first_par.getnext()

        text = "\n".join(texts)
        return text, skipped

    def __find_start(self, paragraphs: list) -> NavigableString:
        """
        Finds and returns the starting paragraph in a judgment.
        Parameters:
        -----------
        paragraphs: list
            a list of 'p' elements from the judgment soup
        Returns:
        --------
        start:
            the judgment start
        """
        for i, p_ in enumerate(paragraphs):
            if p_.text.strip() == "Judgment" or p_.text.strip() == "Arrêt":
                # The following in the new HTML version
                if (
                    p_.has_attr("class")
                    and p_["class"][0] == "coj-sum-title-1"
                    or p_["class"][0] == "C75Debutdesmotifs"
                ):
                    start = p_.find_next("tr")
                    return start
                nex = p_.find_next_sibling("p")
                while len(nex.text.strip()) == 0:
                    nex = p_.find_next_sibling("p")
                    p_ = nex
                start = nex
            else:
                continue
        return start

    def _isolate_judgment(self):
        """Isolate judgment paragraphs from the HTML soup. Use when no judgment text is available in the XML"""
        soup = self._soup
        paragraphs = soup.find_all("p")
        start = self.__find_start(paragraphs)

        judgment_pars = [start]
        cur = start
        while cur.find_next_sibling():
            try:
                judgment_pars.append(cur.get_text())
                cur = cur.find_next_sibling()
            except AttributeError:
                break

        return judgment_pars

    def _detect_total_paragraphs(self, num_patt):
        grounds_tree = self.extract_grounds()
        if grounds_tree is not None:
            paragraphs = grounds_tree.findall("p")
            p = paragraphs[-1]

            texts = []
            while re.search(num_patt, p.text.strip()) is None:
                texts.append(p.text.strip())
                p = p.getprevious()

            # Append the text that cause the while condition to fail because it start with a number
            texts.append(p.text.strip())
            texts.reverse()

            text = "\n".join(texts)
            parno = re.search(num_patt, p.text.strip()).group("number")
            return int(parno), text
        else:
            # working with html
            paragraphs = self.soup.find_all("p")

            # patt = self.select_patt_type(paragraphs[0])  # use that instead of num_patt

            for p in reversed(paragraphs):
                num = re.search(num_patt, p.text)
                if num:
                    return int(num.group("number")), p.text
                else:
                    continue
            return None

    @staticmethod
    def tree_xpath_regex(tree, regex, following=False, following_no=None):
        """Runs an XPath with text regular expression on an etree
        :param tree:
        :param regex: the pregex
        Parameters
        ----------
        tree : etree
            the etree to run XPath with regex
        regex : str
            a string with the regex to run on. This generally should be a regex to detect a number in the
            beginning of a paragraph

        Returns
        -------
        list of lxml.Element or None:
            The element(s) captured by the expression
        """
        ns = {"re": "http://exslt.org/regular-expressions"}
        if not following:
            xpath = "//p[re:test(., '{patt}', 'i')]".format(patt=regex)
        else:
            xpath = "//p[re:test(., '{patt}', 'i')]/following::p[re:test(., '{follow_patt}', 'i')]".format(
                patt=regex, follow_patt=following_no
            )
        results = tree.xpath(xpath, namespaces=ns)
        return results

    def _extractorPreconditions(self, par_num):
        """
        Probably deprecated
        :param par_num:
        :return:
        """
        assert self._version >= 0, "Need to run ECJProcessor.detectVersion(), first"

        if par_num <= 0:
            raise ValueError("Invalid value for paragraph number")


class ECJProcessor15(ECJProcessor):
    """
    Class to process texts from the mid 00's and onwards
    """

    def __init__(self, path):

        super(ECJProcessor15, self).__init__(path)

        if not os.path.exists(self._path):
            # If the path does not exist we should report it as error
            # and better extract the CELEX first, by reading the path parameter
            celex = celex_from_path(self._path)
            logging.error(f"{celex} does not exist")
            print(f"{celex} does not exist")
            raise ValueError(f"File {self._path} does not exist")

        # self._xpath = '//p[starts-with(@id, "point")]//ancestor::tr//text()'
        # elif first_par.find(attrs={'class': 'coj-count'}): # This is used in the latest texts but we can go on
        # with the xpath
        #    return ECJProcessor.NUM_PATT_NO_DOT
        self._tree = None
        self._paragraphs = {}
        self._parno = 0

        with open(self._path, encoding="utf-8") as html:
            self._soup = bs(html.read(), "lxml")
            soup = self._soup
            # newer judgment have paragraphs of the form '<p class="coj-count" id="point35">35</p>', where the number
            # follows the point in the 'id' attribute
            self._paragraphs = soup.find_all("p", attrs={"id": re.compile("point\\d")})
            if len(self._paragraphs) == 0:  # run 2nd style alternative
                self._paragraphs = soup.find_all(
                    "p", attrs={"class": "C01PointnumeroteAltN"}
                )

        self._version = None

    def _by_lxml(self):
        return self._paragraphs

    def _compute_paragraph_number(self):
        # soup = self._soup
        # pars = soup.find_all('p', attrs={'class': 'coj-count'})
        # if len(pars) == 0:
        #     pars = soup.find_all('p', attrs={'class': 'count'})

        # no = int(self._paragraphs[-1].text.split()[0])
        # self._parno = no
        self._parno = len(self._paragraphs)

    @property
    def paragraph_number(self):
        return len(self._paragraphs)

    def detect_version(self):
        """For ECJProcessor15 detect_version is not used so we decided an empty implementation to get rid of
        some encoding trouble that made little sense to debug"""
        return

    def paragraph_extractor(self, par_num):
        """
        Extract paragraph number `par_num`
        NOTE: This is deprecated in ECJProcessor15, because the paragraphs are already extracted at the HTML
          and the constructor populates self._paragraphs
        """
        self._extractorPreconditions(par_num)
        soup = self._soup

        # BeautifulSoup trick: find the paragraphs that match the given
        # paragraph number

        pars = self._paragraphs

        # NOTE: why can't we finish here
        if par_num > len(pars) - 1:
            # For some reason BeautifulSoup breaks on very long documents
            # and only parses up to a certain paragraph.
            # I therefore use lxml
            pars = self._by_lxml()
            text = etree.tounicode(pars[par_num - 1], method="text")
            return text.strip()

        par = pars[par_num - 1]
        next_par = pars[par_num]

        text = ""
        # iterate over the next paragraphs until you find the paragraph with the next paragraph number
        # or the end of the document
        while par != next_par:
            try:
                text += par.text + "\n"
                par = par.find_next("p")
            except AttributeError:
                # Not a real exception. All the input has been consumed so return
                break
        return text

    def _detect_total_paragraphs(self):
        return len(self._paragraphs)

    def read_paragraphs(self):
        """
        Read the paragraphs of the case passed as input

        Extraction was done by using the fulltext of the judgment passed together
        with XML metadata. This has stopped by the Court and we have to parse the HTML

        Returns
        -------
        paragraphs: dict
            A dictionary storing paragraph contents, indexed by paragraph number
        """

        paragraphs = {}

        for i, p in enumerate(self._paragraphs, start=1):
            if p.a:
                p_text = p.text.strip()
                # p_number = re.search('^\\d+', p_text)
                p_text = re.sub("^\\d+", "", p_text)
            else:
                p_text = p.parent.parent.text
            # Paragraph number is currently the current index of the `paragraphs` array -1.
            # We can also do parno = int(p_text.split()[0])
            paragraphs[i] = p_text.strip()  # just remove whitespace before and after

        self._paragraphs = paragraphs

        return paragraphs

    def celex_from_path(self):
        casedir = os.path.dirname(self._path)
        # The following works as long as the paths to judgment.html end up in "<celex>/EN/judgment.html"
        celex = os.path.split(os.path.split(casedir)[0])[1]

        return celex


class ECJTextProcessor(ECJProcessor15):
    JUDGMENT_START = r"gives the following(\n)+Judgment"

    def __init__(self, path):
        super().__init__(path)

    def extract_paragraphs(self, judgment_paragraphs):
        pass

    def detect_version(self):
        pass

    def paragraph_extractor(self, par_num, num_pattern):
        pass

    def __find_start(self, paragraphs: list) -> NavigableString:
        pass

    def _isolate_judgment(self):
        pass

    def _detect_total_paragraphs(self, num_patt):
        pass

    @staticmethod
    def tree_xpath_regex(tree, regex, following=False, following_no=None):
        pass

    def _extractorPreconditions(self, par_num):
        pass

    def read_paragraphs(self):
        text = self._soup.get_text(separator="\n")
        paragraphs = {}

        current_paragraph_number = None
        current_paragraph_text = ""

        # Find the end of the preamble (start of judgment text) using re.search
        preamble_end = re.search(self.JUDGMENT_START, text)
        if preamble_end:
            text = text[preamble_end.end() :].strip()

        # Split the text into lines
        lines = text.splitlines()

        for line in lines:
            # Check if the line starts with a paragraph number
            match = re.match(r"^(\d+)$", line)
            if match:
                # If we were already processing a paragraph, store it
                if current_paragraph_number is not None:
                    paragraphs[current_paragraph_number] = (
                        current_paragraph_text.strip()
                    )

                # Start a new paragraph
                current_paragraph_number = int(match.group(1))
                current_paragraph_text = line
            else:
                # Append the line to the current paragraph text
                current_paragraph_text += " " + line + "\n"

        # Store the last paragraph
        if current_paragraph_number is not None:
            paragraphs[current_paragraph_number] = current_paragraph_text.strip()

        self._paragraphs = paragraphs

        return paragraphs


if __name__ == "__main__":
    parser = ECJProcessor15("judgments/62011CJ0667/eng_judgment.html")
    print(len(parser.read_paragraphs()))
