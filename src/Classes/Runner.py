# Author : Florian Picca <florian.picca@oppida.fr>
# Date : February 2020

from . import bsiTests
from . import nistTests
from . import prngs
from ..utils import *


class Runner:
    """
    Class that runs the tests, prints their results and handle the random pool.
    """

    def __init__(self, prng):
        self.prng = prng
        if type(self.prng) is prngs.External:
            # Init the pool with all the content of the file.
            self.pool = self.prng.pool
        else:
            # Init the pool with dynamically generated random data
            # At least 5 140 000 bits are needed to run the BSI procedures.
            self.pool = self.prng.getBits(10000000)

    def _getFromPool(self, n):
        """
        Will read n bits of data from the random pool.
        This is used so the same random data is used for multiple tests and not regenerated each time.
        """
        bits = self.pool[:n]
        if len(bits) != n:
            return None
        return bits

    def _runTest(self, test, testType, *args):
        """
        Run a generic individual test and prints the result
        """
        print("Running {} test : ".format(test.name.replace("_", " ")), end="")
        if testType == TEST_TYPES.BSI:
            if bsiTests.TEST_FUNC[test](*args):
                psuccess("SUCCESS")
            else:
                perror("FAIL")
        elif testType == TEST_TYPES.NIST:
            if nistTests.TEST_FUNC[test](*args):
                psuccess("SUCCESS")
            else:
                perror("FAIL")
        else:
            perror("Unknown test type : {}".format(testType.name))

    def singleBSITest(self, test):
        """
        Runs a single BSI test and handles all the error cases.
        """
        n = bsiTests.MIN_LENGTHS.get(test)
        if n is None:
            raise NotImplementedError("No minimum bit length defined for : {}".format(test.name))

        bits = self._getFromPool(n)
        if bits is None:
            pwarning("Not enough bits to perform this test.")
            return None

        if bsiTests.TEST_FUNC.get(test) is None:
            raise NotImplementedError("No test function associated to : {}".format(test.name))

        self._runTest(test, TEST_TYPES.BSI, bits)

    def singleNISTTest(self, test):
        """
        Runs a single NIST test and handles all the error cases.
        """
        n = nistTests.MIN_LENGTHS.get(test)
        if n is None:
            raise NotImplementedError("No minimum bit length defined for : {}".format(test.name))

        # NIST tests are better with more bits, use the whole pool
        bits = self.pool
        if len(bits) < n:
            pwarning("Not enough bits to perform this test.")
            return None

        if nistTests.TEST_FUNC.get(test) is None:
            raise NotImplementedError("No test function associated to : {}".format(test.name))

        self._runTest(test, TEST_TYPES.NIST, bits)

    def runBSITests(self, window):
        """
        Runs the standard BSI procedures A and B.
        """

        pbar = window.window["-pbar-"]
        pbar2 = window.window["-pbar2-"]
        count = 0
        pbar.update(max=2, current_count=count, visible=True)
        pbar2.update(visible=True)

        print("Running procedure A from the BSI : ")
        if bsiTests.procedure_A(self._getFromPool, window):
            psuccess("SUCCESS")
        else:
            perror("FAIL")
        
        count += 1
        pbar.update(count)
        window.window.refresh()

        print("Running procedure B from the BSI : ")
        if bsiTests.procedure_B(self._getFromPool, window):
            psuccess("SUCCESS")
        else:
            perror("FAIL")

        count += 1
        pbar.update(count)
        window.window.refresh()
        # purely visual
        # let the time for the progress bar to fill up completely before removing it
        import time
        time.sleep(0.1)
        pbar.update(current_count=0, visible=False)
        pbar2.update(current_count=0, visible=False)

    def runNISTTests(self, window):
        """
        TODO
        """
        print("NIST test procedures TODO")
