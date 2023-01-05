import markdownify
import re


def sanitize_markdown(markdown_text: str) -> str:
    """
    Sanitize markdown text by:
     - removing all images references
     - some {% %} directives
     - replacing some unicode characters (to reduce number of tokens required)
     - removing 3 or more hyphens
     - replacing triple or more newlines to two, 
     - replacing double or more whitespace chars to one space
    """

    # Remove all image references "![..](..)"
    markdown_text = re.sub(
        # r"!\[(?<alttext>[^\]]*)\]\((?<filename>.*?)(?=\"|\))(?<optionalpart>\".*\")?\)"
        r"(?m)!\[([^\]]*)\]\((.*?)(?=\"|\))(\".*\")?\)", "",
        markdown_text)

    # Remove "cover:. .."" or "coverY: ...""  at line start
    markdown_text = re.sub(r"(?m)^coverY?\:.*\n", "",
                           markdown_text)

    # Replace 3 or more hyphens with empty string
    #   loads of hyphens are used in tables , lot of tokens w/o use
    markdown_text = re.sub(r"-{3,}", "", markdown_text, flags=re.MULTILINE)

    # Replace "{% hint .... %}" with empty string
    markdown_text = re.sub(r"(?m){% ?hint.*}\n?", "", markdown_text)

    # Replace "{% endhint %}" with empty string
    markdown_text = re.sub(r"(?m){% ?endhint ?%}\n?", "", markdown_text)

    # Replace "{% tabs %}" with empty string
    markdown_text = re.sub(r"(?m){% ?tabs.*}\n?", "", markdown_text)

    # Replace "{% tab title = "..." %}" with empty string but keep title
    tab_titles = re.findall(
        r"({% *tab *title *= *(\"|')(.*?)\2 * %})", markdown_text)
    for match in tab_titles:
        markdown_text = markdown_text.replace(
            match[0], "## " + match[2] + "\n")

    # Replace "{$ embed url = "..." %}" with empty string but keep url
    embed_urls = re.findall(
        r"({% *embed *url *= *(\"|')(.*?)\2 * %})", markdown_text)
    for match in embed_urls:
        markdown_text = markdown_text.replace(
            match[0], "* " + match[2])

    # Replace "{% endtab %}" and endtabs with empty string
    markdown_text = re.sub(r"(?m){% ?endtab.*%}\n?", "", markdown_text)

    # replace triple or more newlines to two
    markdown_text = re.sub(r"\n{3,}", "\n\n", markdown_text)

    # replace double or more whitespace chars to one space
    markdown_text = re.sub(r"[^\S\r\n]{2,}", " ", markdown_text)

    # replace unicode characters (–  ” “ ’) to non unicode equivalent (less tokens)
    for old, new in {"–": "-", "—": "-", "”": "\"", "“": "\"", "’": "'", "‘": "'"}.items():

        markdown_text = markdown_text.replace(old, new)

    for character in markdown_text:
        if ord(character) > 127:  # ASCII characters have a value less than 128
            print(
                f"UTF8 character in input text: '{character}' {character.encode('ascii', 'namereplace')}")

    return markdown_text


def convert_html_in_text(content: str, tag: str) -> str:

    matches = re.findall(
        rf"<{tag}.*?>[\s\S]*?<\/{tag}>", content, flags=re.MULTILINE)

    for match in matches:
        md = markdownify.markdownify(match, heading_style="ATX")
        content = content.replace(match, md)
    return content


def get_title(section_content: str, level=1) -> str:
    lev = str(level)
    section_title = re.search(
        r"(?m)(?<=^#{" + lev + r"} ).*$", section_content).group(0).strip()

    if section_title and len(section_title) > 0:
        return section_title
    else:
        return None


def remove_title(section_content: str) -> str:
    section_lines = section_content.splitlines()

    section_without_title = '\n'.join(section_lines[1:]).strip()
    return section_without_title
