# Author : Florian Picca <florian.picca@oppida.fr>
# Date : February 2020

from enum import Enum

TEST_TYPES = Enum("Test_Type", "BSI NIST")
BSI_TESTS = Enum("BSI_Tests", "disjointness monobit poker runs longrun autocorrelation "
                              "uniform_distribution homogeneity entropy")
NIST_TESTS = Enum("NIST_Tests", "monobit frequency_block runs longestrun non_overlapping_template "
                                "overlapping_template cusum binary_matrix_rank linear_complexity serial "
                                "approximate_entropy random_excursion random_excursion_variant maurer spectral")
PRNGS = Enum("PRNGS", "External RANDU PyRandom DevURandom")
PARSERS = Enum("Parser", "Raw Hex Binary")


def strToEnum(e, s):
    """
    Converts a string representing an enum value, into an enum type.

    :param e: The enum from which the string represents a value.
    :type e: Enum
    :param s: The string representation of the value.
    :type s: str
    :return: The enum whose value is that string.
    :rtype: Enum

    **Examples** ::

        >>> Lib = Enum("Lib", "Test OpenSSL")
        >>> strToEnum(Lib, "OpenSSL")
        <Lib.OpenSSL: 2>

    """
    try:
        return e[s]
    except KeyError:
        return None


def perror(m):
    """
    Prints a message in red color.

    :param    m: The message to print.
    :type     m: str

    **Examples** ::

        >>> perror("Error")
        Error
    """
    import PySimpleGUI as psg
    psg.cprint(m, colors='red')


def psuccess(m):
    """
    Prints a message in green color.

    :param    m: The message to print.
    :type     m: str

    **Examples** ::

        >>> psuccess("Success")
        Success
    """
    import PySimpleGUI as psg
    psg.cprint(m, colors='green3')


def pwarning(m):
    """
    Prints a message in yellow color.

    :param    m: The message to print.
    :type     m: str

    **Examples** ::

        >>> pwarning("Warning")
        Warning
    """
    import PySimpleGUI as psg
    psg.cprint(m, colors='orange')


def berlekamp_massey(data):
    """
    Finds the minimal LFSR and minimal polynomial to generate the given data.

    :param data: list of 1s and 0s.
    :type data: list[int]

    :return: The linear complexity
    :rtype: int
    """
    import numpy as np

    n = len(data)
    c_x, b_x = np.zeros(n, dtype=np.int), np.zeros(n, dtype=np.int)
    c_x[0], b_x[0] = 1, 1
    l, m, i = 0, -1, 0
    while i < n:
        v = data[(i - l):i]
        v = v[::-1]
        cc = c_x[1:l + 1]
        delta = (data[i] + np.dot(v, cc)) % 2
        if delta == 1:
            temp = c_x[:]
            p = np.zeros(n, dtype=np.int)
            for j in range(0, l):
                if b_x[j] == 1:
                    p[j + i - m] = 1
            c_x = (c_x + p) % 2
            if l <= 0.5 * i:
                l = i + 1 - l
                m = i
                b_x = temp
        i += 1
    return l


def rankMat(A):
    """
    Computes the rank of a matrix. numpy.linealg.matrix_rank gives a false result
    """
    n = len(A[0])
    rank = 0
    for col in range(n):
        j = 0
        rows = []
        while j < len(A):
            if A[j][col] == 1:
                rows += [j]
            j += 1
        if len(rows) >= 1:
            for c in range(1, len(rows)):
                for k in range(n):
                    A[rows[c]][k] = (A[rows[c]][k] + A[rows[0]][k]) % 2
            A.pop(rows[0])
            rank += 1
    for row in A:
        if sum(row) > 0:
            rank += 1
    return rank


def isAperiodic(t):
    """
    Determines if a template is aperiodic
    """
    m = len(t)
    for k in range(1, m):
        c = t[m-k:m]
        d = t[:k]
        if c == d:
            return False
    return True
