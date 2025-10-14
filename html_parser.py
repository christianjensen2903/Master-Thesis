import logging
import sys

from bs4 import BeautifulSoup as bs
from bs4 import NavigableString
import re
from lxml import etree  # type: ignore
import os, os.path
import random
import glob


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
        # this is unfortunately caused when we group by CELEX and we have double directories for the same case, e.g.
        # C-105_04 and C-105_04_P
        if "," in self._path:
            print("Comma in path")
            self._path = self._path.split(",")[0]
            print(self._path)

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
        pass

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

        # Classes that should stop paragraph collection:
        # - Titles: C04Titre1, S40Titre
        # - Dispositif (operative part): C41DispositifIntroduction, C30Dispositifalinea
        # - Metadata: C77Signatures, C42FootnoteLangue
        stop_classes = [
            "C04Titre1",
            "S40Titre",
            "C41DispositifIntroduction",
            "C30Dispositifalinea",
            "C77Signatures",
            "C42FootnoteLangue",
        ]

        while cur.find_next_sibling("p", attrs={"class": "C01PointnumeroteAltN"}):
            try:
                # Collect the main paragraph text and all following sibling paragraphs
                # until we hit the next C01PointnumeroteAltN or a title
                par_texts = [" ".join(list(cur.stripped_strings))]

                # Get all siblings until the next main paragraph or stop class
                next_sibling = cur.find_next_sibling("p")
                while next_sibling:
                    sibling_classes = next_sibling.get("class", [])
                    # Stop if we hit a main paragraph or any stop class
                    if "C01PointnumeroteAltN" in sibling_classes or any(
                        sc in sibling_classes for sc in stop_classes
                    ):
                        break
                    sibling_text = " ".join(list(next_sibling.stripped_strings))
                    if len(sibling_text) > 0:
                        par_texts.append(sibling_text)
                    next_sibling = next_sibling.find_next_sibling("p")

                # Combine all parts into one paragraph
                par_text = " ".join(par_texts)
                if len(par_text) > 0:
                    pars.append(par_text)

                # Move to the next main paragraph
                cur = cur.find_next_sibling(
                    "p", attrs={"class": "C01PointnumeroteAltN"}
                )
            except AttributeError:
                break

        # add last element and all its following siblings (until a stop class)
        par_texts = [" ".join(list(cur.stripped_strings))]
        next_sibling = cur.find_next_sibling("p")
        while next_sibling:
            sibling_classes = next_sibling.get("class", [])
            # Stop if we hit any stop class
            if any(sc in sibling_classes for sc in stop_classes):
                break
            sibling_text = " ".join(list(next_sibling.stripped_strings))
            if len(sibling_text) > 0:
                par_texts.append(sibling_text)
            next_sibling = next_sibling.find_next_sibling("p")

        par_text = " ".join(par_texts)
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

        # load the judgment as soup
        # isolate the actual judgment
        judgment_paragraphs = self._isolate_judgment()

        # Try to navigate with BeautifulSoup
        paragraphs = self.extract_paragraphs(judgment_paragraphs)

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
            # print(p_.text.strip())
            if p_.text.strip() != "Judgment" and p_.text.strip() != "Arrêt":
                continue
            # The following in the new HTML version
            # if (
            #     p_.has_attr("class")
            #     and p_["class"][0] == "coj-sum-title-1"
            #     or p_["class"][0] == "C75Debutdesmotifs"
            # ):
            #     start = p_.find_next("tr")
            #     return start
            nex = p_.find_next_sibling("p")
            while len(nex.text.strip()) == 0:
                nex = p_.find_next_sibling("p")
                p_ = nex
            start = nex

        return start

    def _isolate_judgment(self):
        """Isolate judgment paragraphs from the HTML soup"""
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


class ECJProcessor15(ECJProcessor):
    """
    Class to process texts from the mid 00's and onwards
    """

    def __init__(self, path):

        super(ECJProcessor15, self).__init__(path)

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

    def __find_start(self, paragraphs: list) -> NavigableString:
        return None  # type: ignore

    def _isolate_judgment(self):
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
    # Get all HTML files from the cases folder
    case_files = glob.glob("cases/*.html")

    if not case_files:
        print("No case files found in the cases folder.")
        sys.exit(1)

    # Randomly select a case file
    # random_case = random.choice(case_files)
    random_case = "cases/61972CJ0077.html"
    print(f"Processing random case: {random_case}\n")

    parser = ECJProcessor(random_case)
    paragraphs = parser.read_paragraphs()
    for number, text in paragraphs.items():
        print(f"{number}:")
        print(text)
        print("\n" + "=" * 100 + "\n")
