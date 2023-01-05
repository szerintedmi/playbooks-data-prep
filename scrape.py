from pprint import pprint
import streamlit as st
import re
import copy
import pandas as pd
import os
import fnmatch
import markdownutils as mdutils
from transformers import GPT2TokenizerFast
import numpy as np
from slugify import slugify

st.set_page_config(layout="wide")

MAX_SECTION_DEPTH = 4

HTML_TAGS_TO_CONVERT = ["html", "table", "ul", "ol", "p"]

tokenizer = GPT2TokenizerFast.from_pretrained("gpt2")

test_text = \
    """# Title 1
## Title 1.1
### Title 1.1.1
#### Title 1.1.1.1

text #

# Title 2 - A
subtext title2 A
## Title 2.1
subtext title2.1
### Title 2.1.1
subtext title2.1
#### Title 2.1.1.1
## Title 2.2

"""


def count_tokens(text: str) -> int:
    # same: len(tokenizer(text)['input_ids'])
    return len(tokenizer.encode(text))


def calculate_navigation_info(source_path: str, section_path: list[str]) \
        -> dict[{str, int, str, str, list[str], str, list[str]}]:
    """
    Calculate the navigation info from sourcePath and sectionPath (black magic)
        - playbookUrl
        - pageTitle
        - subtitles
        - anchorSlug: to link to the section in the page
        - pathDepth: url file path depth
        - sectionPath: the full section path (including dummy subtitle as empty string if a section level was missing)
    """

    path_depth = source_path.count("/") - 3

    playbook_url = source_path.replace("./content/", "https://", 1)
    playbook_url = playbook_url.replace("-playbook/", ".playbook.ee/", 1)
    playbook_url = playbook_url.replace(
        "README.md", "", 1)  # it's the root

    # SUMMARY.md is the Toc in the left panel, fine to link to the containing folder:
    playbook_url = playbook_url.replace("SUMMARY.md", "", 1)
    playbook_url = playbook_url.replace(".md", "", 1)

    page_title = section_path[0]

    subtitles = list(filter(None, section_path[1:]))

    anchor_slug = ""
    if len(section_path) > 1:
        anchor_slug = "#" + \
            slugify(section_path[-1],
                    replacements=([["&", "and"]]))[:100]

    navInfo = {'playbookUrl': playbook_url, 'pathDepth': path_depth,
               'pageTitle': page_title, 'subTitles': subtitles, 'anchorSlug': anchor_slug, 'sectionPath': section_path}

    return navInfo


def extract_sections(content: str, level: int, path: [str] = []) -> str:
    """recursively  extract every section to flat structure"""
    sections = []
    current_path = copy.deepcopy(path)

    lev = str(level)

    # text before the first title on the given level for text before #(s) or markdown where eg. ### comes right after #)
    section_before_title = re.search(
        r"(?sm)\A.*?(?=^#{" + lev + "} |\Z)", content)  # (?sm)\A.*?(?=^#{1} |\Z)
    if section_before_title:
        pre_title_content = section_before_title.group(0).strip()
        if len(pre_title_content) > 0:
            # sections.append(
            #     {"level": level, "path": [*current_path, "XX"], "content": pre_title_content})

            if level <= MAX_SECTION_DEPTH:
                subsections = extract_sections(
                    pre_title_content, level + 1, [*current_path, ""])
                sections.extend(subsections)

    # Sections by titles at the given level
    matches = re.findall(
        r"(?sm)^#{" + lev + r"} .*?(?=^#{1," + str(level) + "} |\Z)", content)

    for match in matches:
        section_content = match.strip()

        current_path = copy.deepcopy(path)

        section_title = mdutils.get_title(match, level)
        if section_title:
            current_path.append(section_title)

        section_without_title = mdutils.remove_title(section_content)

        if len(section_without_title) > 0:
            sections.append(
                {"level": level, "sectionPath": current_path, "contentLength": len(section_without_title), "tokenCount": count_tokens(section_without_title), "content": section_without_title})

            if level <= MAX_SECTION_DEPTH:
                subsections = extract_sections(match, level + 1, current_path)
                sections.extend(subsections)

    return sections


def process_md_files(root_dir: str, playbook_name: str):
    """
    Iterate over all the md files in the root directory and its subdirectories
              extract sections, calculate token_len and populate DataFrame
    """

    df = pd.DataFrame()

    for root, dirs, files in os.walk(root_dir):
        # extract the folder title from README.md
        folder_title = ""
        readme_file_path = os.path.join(root, "README.md")
        if os.path.exists(readme_file_path):
            with open(readme_file_path, "rb") as f:
                content = f.read().decode()
                folder_title = mdutils.get_title(content)

        files = [f for f in files if fnmatch.fnmatch(f, "*.md")]
        for file in files:
            file_path = os.path.join(root, file)

            with open(file_path, "rb") as f:
                content = f.read().decode()

                # convert text in between html tags to markdown
                for tag in HTML_TAGS_TO_CONVERT:
                    content = mdutils.convert_html_in_text(content, tag)

                content = mdutils.sanitize_markdown(content)

                # for pages in the root and  READMEs anywhere the title in the content ("# ...") defines the place
                #    for other pages we need to include folder title (i.e. the left navigation location) in sectionPath
                section_path = []
                path_depth = file_path.count("/") - 3
                if path_depth != 0 and not file_path.endswith("README.md"):
                    section_path = [folder_title]

                sections = extract_sections(content, 1, section_path)

                if len(sections) > 0:
                    df_content = pd.DataFrame(sections)

                    df_content['navInfo'] = df_content.apply(
                        lambda x: calculate_navigation_info(file_path, x["sectionPath"]), axis=1)

                    df_content['fullTitle'] = df_content.apply(
                        lambda x: f"{playbook_name} Playbook: " + f" {' - '.join( x.navInfo['subTitles'])}", axis=1)

                    df_content["sourcePath"] = file_path

                    df = pd.concat([df, df_content], ignore_index=True)

    return df


df = pd.DataFrame(
    {'playbookTitle': np.ndarray((0,), dtype=str),
     'fullTitle': np.ndarray((0,), dtype=str),
     'sourcePath': np.ndarray((0,), dtype=str),
     'level': np.ndarray((0,), dtype=int),
     'navInfo': np.ndarray((0,), dtype=object),
     'contentLength': np.ndarray((0,), dtype=int),
     'tokenCount': np.ndarray((0,), dtype=int),
     'content': np.ndarray((0,), dtype=str),
     }
)

md_files = [
    {"path": "./content/advice-process-playbook",  "title": "Advice Process"},
    {"path": "./content/chaos-day-playbook", "title": "Chaos Day"},
    {"path": "./content/digital-platform-playbook",  "title": "Digital Platform"},
    {"path": "./content/inception-playbook",  "title": "Inception"},
    {"path": "./content/remote-working-playbook",  "title": "Remote Working"},
    {"path": "./content/secure-delivery-playbook",  "title": "Secure Delivery"},
    {"path": "./content/you-build-it-you-run-it-playbook",
        "title": "You Build It, You Run It (YBIYRI)"}
]

for f in md_files:
    res = process_md_files(f["path"], f["title"])

    df_res = pd.DataFrame(res)
    df_res["playbookTitle"] = f["title"]

    df = pd.concat([df, res])

df.index = pd.RangeIndex(len(df.index))
df.to_parquet('./results/flattened_content.parquet')

st.write("# Processed ")
st.write(df)

st.write("# Convert html ")
conv_col1, conv_col2 = st.columns(2)
html_text = conv_col1.text_area("markdown with html:", height=300)
converted_text = mdutils.convert_html_in_text(html_text, HTML_TAGS_TO_CONVERT)
conv_col2.write(converted_text)

st.write("# Sanitize html ")
sanit_col1, sanit_col2 = st.columns(2)
text2sanitize = sanit_col1.text_area("markdown to sanitize:", height=300)
sanitized_text = mdutils.sanitize_markdown(text2sanitize)
sanit_col2.write(sanitized_text)


st.write("# Markdown to sections")
col1, col2 = st.columns(2)
section_level = col2.select_slider(
    "Level to extract: (number of # tags)", options=[1, 2, 3, 4, 5],)
markdown_text = col1.text_area("MarkDown text:", height=300, value=test_text)
markdown_sections = extract_sections(markdown_text, section_level)
col2.write(markdown_sections)


print("DONE.\n\n")
