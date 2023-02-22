import unicodedata
import re


def string_cleaning(str_name: str, bln_nospace: bool = False):
    """
    Method that cleans fields of spaces, punctuation and strange characters
    :param str_name: the string to be treated
    :param bln_nospace: if True, spaces will be removed (default: True)
    :return: the cleaned string
    """
    if bln_nospace:
        str_clean = re.sub('\\W+', '', str(str_name)).upper() 
    else:
        lst_clean = re.split(r"[\,\s]+", str(str_name))
        lst_clean = ' '.join(lst_clean).split()
        str_clean = ' '.join([re.sub('\\W+', '', str(x)) for x in lst_clean])
        str_clean = re.sub(r"^\s+$", '', str(str_clean)).upper()
    return ''.join(c for c in unicodedata.normalize('NFD', str_clean)
                   if unicodedata.category(c) != 'Mn')


def string_cleaning_full(str_name: str):
    """
    Method that cleans the string of all special characters (including spaces, numbers, accents and German characters)
    :param str_name: the string to be treated
    :return: the cleaned string
    """
    str_clean = re.sub('\\W+|\\d+', '', str(str_name)).upper()
    str_clean = unicodedata.normalize('NFKD', str_clean).encode('ASCII', 'ignore')
    return ''.join(c for c in unicodedata.normalize('NFD', str_clean.decode("utf-8"))
                   if unicodedata.category(c) != 'Mn')


def memoize(func):
    """
    Support method for the calculation of the Levenshtein distance, uses a dictionary "memo" to store
    the function results.
    :param func: takes a function as an argument
    :return: a reference to the helper function
    """
    dct_mem = {}

    def memoizer(*args, **kwargs):
        key = str(args) + str(kwargs)
        if key not in dct_mem:
            dct_mem[key] = func(*args, **kwargs)
        return dct_mem[key]

    return memoizer

@memoize
def levenshtein(str_one, str_two):
    """
    Method that calculates the Levenshtein distance between two strings
    :param str_one: string number one
    :param str_two: string number two
    :return: an integer containing Levenshtein's distance
    """
    if str_one == "":
        return len(str_two)
    if str_two == "":
        return len(str_one)
    if str_one[-1] == str_two[-1]:
        cost = 0
    else:
        cost = 1

    int_res = min([levenshtein(str_one[:-1], str_two) + 1,
                   levenshtein(str_one, str_two[:-1]) + 1,
                   levenshtein(str_one[:-1], str_two[:-1]) + cost])

    return int_res


def remove_group(str_name: str):
    return re.sub(r'GRUPPO\b\s+', "", str_name)