# Author : Florian Picca <florian.picca@oppida.fr>
# Date : February 2020

from ..utils import *
from binascii import hexlify


class RawParser:
    """
    Converts a raw binary file into a string of 1s and 0s.
    """
    def __init__(self, path):
        self.path = path

    def convert(self):
        try:
            with open(self.path, "rb") as f:
                c = f.read()
                n = len(c) * 8
                c = hexlify(c)
                c = int(c, 16)
                bits = bin(c)[2:].zfill(n)
                assert set(bits) == set("01"), "Invalid characters found : {}".format(set(bits))
            return bits
        except Exception as e:
            print(e)
            return None


class HexParser:
    """
    Converts a file containing hexadecimal characters into a string of 1s and 0s.
    """

    def __init__(self, path):
        self.path = path

    def convert(self):
        try:
            with open(self.path, "rb") as f:
                c = f.read()
                # removes special chars
                c = c.replace(b" ", b"").replace(b'\n', b'').replace(b'\t', b'')
                n = len(c) * 4
                c = int(c, 16)
                bits = bin(c)[2:].zfill(n)
                assert set(bits) == set("01"), "Invalid characters found : {}".format(set(bits))
            return bits
        except Exception as e:
            print(e)
            return None


class BinaryParser:
    """
    Converts a file containing 0s and 1s (on multiples lines or not) into a string of 1s and 0s.
    """

    def __init__(self, path):
        self.path = path

    def convert(self):
        try:
            with open(self.path, "rb") as f:
                c = f.read()
                # removes special chars
                bits = c.replace(b" ", b"").replace(b'\n', b'').replace(b'\t', b'')
                bits = bits.decode()
                assert set(bits) == set("01"), "Invalid characters found : {}".format(set(bits))
            return bits
        except Exception as e:
            print(e)
            return None


# -------------
# Mappings
# -------------
PARSER_CLASS = {
    PARSERS.Raw: RawParser,
    PARSERS.Hex: HexParser,
    PARSERS.Binary: BinaryParser,
}
