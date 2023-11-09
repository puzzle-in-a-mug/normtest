### GENERAL ###
def empty_spaces(fields, field):
    return len(max(fields, key=len)) - len(str(field))


### TECHREPORT ###
TECHREPORT_REQUIRED_TEMPLATE = """@techreport{{{citekey},
  author      = {{{author}}},
  title       = {{{title}}},
  institution = {{{institution}}},
  year        = {{{year}}}
}}"""


def make_techreport(citekey, author, title, institution, year, export=False):
    bib = TECHREPORT_REQUIRED_TEMPLATE.format(
        **{
            "citekey": citekey,
            "author": author,
            "title": title,
            "institution": institution,
            "year": year,
        }
    )
    if export:
        with open(f"`{citekey}`.bib", "w") as my_bib:
            my_bib.write(bib)

    return bib


### ARTICLE ###

OPTIONAL_FIELDS_ARTICLE = (
    "translator",
    "annotator",
    "commentator",
    "subtitle",
    "titleaddon",
    "editor",
    "editora",
    "editorb",
    "editorc",
    "journalsubtitle",
    "journaltitleaddon",
    "issuetitle",
    "issuesubtitle",
    "issuetitleaddon",
    "language",
    "origlanguage",
    "series",
    "volume",
    "number",
    "eid",
    "issue",
    "month",
    "pages",
    "version",
    "note",
    "issn",
    "addendum",
    "pubstate",
    "doi",
    "eprint",
    "eprintclass",
    "eprinttype",
    "url",
    "urldate",
)


def make_article(
    author,
    title,
    journaltitle,
    citekey,
    year=None,
    date=None,
    export=True,
    **optionals,
):
    """This function generates a `.bib` file for an `@article` entry.

    Parameters
    ----------
    author : str
            The person or persons who wrote the article. The authors' names must be separated by `" and "`;
    title : str
            The name of the article;
    journaltitle : str
            The name of a journal, a newspaper, or some other periodical.
    year : int, optional
            The year of publication; If `year=None`, the `date` parameter must not be `None`;
    date : str, optional
            The publication date. If `date=None`, the `year` parameter must not be `None`;
    citekey : str
            The name that is used to uniquely identify the entry. If None, the algorithm will generate an automatic `citekey` based on the name of the authors and the year. See Notes for details;
    export : bool, optional
            Whether to export the `citekey.bib` file (`True`, default) or not (`False`);
    **optionals : str
            Optional fields for an article entry (`volume`, `number`, `pages`, etc).


    Returns
    -------
    citation : str
            The citation generated
    bib : file
            The `citekey.bib` file (only if `export=True`)


    Notes
    -----


    Examples
    --------



    """

    if year is None and date is None:
        try:
            raise ValueError("FieldNotFoundError")
        except ValueError:
            print("Parameters `year` and `date` cannot be `None` at the same time.\n")
            raise

    field_template = "  {field_tag}{spaces} = {{{tag_value}}},\n"

    fields = {
        "author": author,
        "title": title,
        "journaltitle": journaltitle,
        "year": year,
        "date": date,
    }
    # getting all fields types
    optional_fields = dict.fromkeys(OPTIONAL_FIELDS_ARTICLE)
    optional_fields.update(optionals)
    fields.update(optional_fields)

    filtered = {k: v for k, v in fields.items() if v is not None}
    fields.clear()
    fields.update(filtered)

    # building the citation
    # first line:
    citation = ["@article{{{citekey},\n".format(citekey=citekey)]

    # ading fields
    for key, value in fields.items():
        spaces = " " * empty_spaces(fields.keys(), key)
        citation.append(
            field_template.format(field_tag=key, spaces=spaces, tag_value=value)
        )
    citation[-1] = citation[-1][:-2] + citation[-1][-1:]

    citation.append("}")

    if export:
        with open(f"{citekey}.bib", "w") as my_bib:
            for i in range(len(citation)):
                my_bib.write(f"{citation[i]}")

    return """""".join(citation)
