# Author : Florian Picca <florian.picca@oppida.fr>
# Date : February 2020

from ..utils import *
import time
import numpy


# -------------------
# Individual tests
# -------------------

def disjointness_test(bitstream):
    """
    BSI test T0.
    https://www.bsi.bund.de/SharedDocs/Downloads/DE/BSI/Zertifizierung/Interpretationen/AIS_31_Functionality_classes_evaluation_methodology_for_true_RNG_e.pdf?__blob=publicationFile&v=1
    https://www.bsi.bund.de/SharedDocs/Downloads/DE/BSI/Zertifizierung/Interpretationen/AIS_31_Functionality_classes_for_random_number_generators_e.pdf?__blob=publicationFile&v=1

    :param bitstream: A string of 0s and 1s representing the input random data.
    :type bitstream: str

    :return: True if the Test succeeded, False if it failed and None if there was an error.
    """
    bitLen = 48
    N = 2 ** 16

    # Check if we have sufficient random data
    if len(bitstream) < N * bitLen:
        perror("Not enough random data to perform this test.")
        return None

    # Make an array with all N numbers of bitLen bits
    numbers = []
    for i in range(N):
        # construct a number from the bits
        start = i * bitLen
        n = int(bitstream[start:start + bitLen], 2)
        numbers.append(n)

    # To check for disjointness, simply compare the length of the array against the length of a set
    # Duplicate numbers are removed in a set
    return len(numbers) == len(set(numbers))


def monobit_test(bitstream):
    """
    BSI test T1.
    https://www.bsi.bund.de/SharedDocs/Downloads/DE/BSI/Zertifizierung/Interpretationen/AIS_31_Functionality_classes_evaluation_methodology_for_true_RNG_e.pdf?__blob=publicationFile&v=1
    https://www.bsi.bund.de/SharedDocs/Downloads/DE/BSI/Zertifizierung/Interpretationen/AIS_31_Functionality_classes_for_random_number_generators_e.pdf?__blob=publicationFile&v=1

    :param bitstream: A string of 0s and 1s representing the input random data.
    :type bitstream: str

    :return: True if the Test succeeded, False if it failed and None if there was an error.
    """
    N = 20000

    # Check if we have sufficient random data
    if len(bitstream) < N:
        perror("Not enough random data to perform this test.")
        return None

    x = bitstream[:N].count("1")
    res = 9654 < x < 10346
    if not res:
        print("Got x = {}, Expected : 9654 < x < 10346".format(x))
    return res


def poker_test(bitstream):
    """
    BSI test T2.
    https://www.bsi.bund.de/SharedDocs/Downloads/DE/BSI/Zertifizierung/Interpretationen/AIS_31_Functionality_classes_evaluation_methodology_for_true_RNG_e.pdf?__blob=publicationFile&v=1
    https://www.bsi.bund.de/SharedDocs/Downloads/DE/BSI/Zertifizierung/Interpretationen/AIS_31_Functionality_classes_for_random_number_generators_e.pdf?__blob=publicationFile&v=1

    :param bitstream: A string of 0s and 1s representing the input random data.
    :type bitstream: str

    :return: True if the Test succeeded, False if it failed and None if there was an error.
    """
    N = 5000
    bitLen = 4

    # Check if we have sufficient random data
    if len(bitstream) < N * bitLen:
        perror("Not enough random data to perform this test.")
        return None

    # Count occurrences of numbers of bitLen bits
    occurrences = [0] * (2 ** bitLen)
    for j in range(N):
        # construct a number from the bits
        start = j * bitLen
        cj = int(bitstream[start:start + bitLen], 2)
        occurrences[cj] += 1

    sumFi2 = 0
    for x in occurrences:
        sumFi2 += x ** 2
    T = (((2 ** bitLen) / N) * sumFi2) - N

    res = 1.03 < T < 57.4
    if not res:
        print("Got T = {}, Expected : 1.03 < T < 57.4".format(T))
    return res


def runs_test(bitstream):
    """
    BSI test T3. Not the same as the runs test from the NIST.
    https://www.bsi.bund.de/SharedDocs/Downloads/DE/BSI/Zertifizierung/Interpretationen/AIS_31_Functionality_classes_evaluation_methodology_for_true_RNG_e.pdf?__blob=publicationFile&v=1
    https://www.bsi.bund.de/SharedDocs/Downloads/DE/BSI/Zertifizierung/Interpretationen/AIS_31_Functionality_classes_for_random_number_generators_e.pdf?__blob=publicationFile&v=1

    :param bitstream: A string of 0s and 1s representing the input random data.
    :type bitstream: str

    :return: True if the Test succeeded, False if it failed and None if there was an error.
    """
    N = 20000

    # Check if we have sufficient random data
    if len(bitstream) < N:
        perror("Not enough random data to perform this test.")
        return None

    # Count the occurrences of runs (for 0s and 1s) for size from 1 to 6+
    runs0 = [0] * 6
    runs1 = [0] * 6
    previous = bitstream[0]
    l = 0
    for i in range(1, N):
        bit = bitstream[i]
        if bit == previous:
            # if the run is longer than 6 bits, count it as 6
            if l < 5:
                l += 1
        else:
            if previous == "0":
                runs0[l] += 1
            else:
                runs1[l] += 1
            l = 0
        previous = bit

    # Check against the permitted intervals
    permitted = [(2267, 2733), (1079, 1421), (502, 748), (223, 402), (90, 223), (90, 223)]
    for i in range(5):
        if not (permitted[i][0] <= runs0[i] <= permitted[i][1]):
            print("Got runs0[{}] = {}, Expected : {} < runs0[{}] < {}".format(i, runs0[i], permitted[i][0], i,
                                                                              permitted[i][1]))
            return False
        if not (permitted[i][0] <= runs1[i] <= permitted[i][1]):
            print("Got runs1[{}] = {}, Expected : {} < runs1[{}] < {}".format(i, runs1[i], permitted[i][0], i,
                                                                              permitted[i][1]))
            return False
    return True


def longrun_test(bitstream):
    """
    BSI test T4. Not the same as the longest run test from the NIST.
    https://www.bsi.bund.de/SharedDocs/Downloads/DE/BSI/Zertifizierung/Interpretationen/AIS_31_Functionality_classes_evaluation_methodology_for_true_RNG_e.pdf?__blob=publicationFile&v=1
    https://www.bsi.bund.de/SharedDocs/Downloads/DE/BSI/Zertifizierung/Interpretationen/AIS_31_Functionality_classes_for_random_number_generators_e.pdf?__blob=publicationFile&v=1

    :param bitstream: A string of 0s and 1s representing the input random data.
    :type bitstream: str

    :return: True if the Test succeeded, False if it failed and None if there was an error.
    """
    N = 20000

    # Check if we have sufficient random data
    if len(bitstream) < N:
        perror("Not enough random data to perform this test.")
        return None

    # No runs longer than 33 must be found
    if bitstream[:N].count("0" * 34) != 0:
        print("Expected no longrun of 34 consecutive 0s, found {}".format(bitstream[:N].count("0" * 34)))
        return False
    if bitstream[:N].count("1" * 34) != 0:
        print("Expected no longrun of 34 consecutive 1s, found {}".format(bitstream[:N].count("1" * 34)))
        return False
    return True


def autocorrelation_test(bitstream):
    """
    BSI test T5.
    https://www.bsi.bund.de/SharedDocs/Downloads/DE/BSI/Zertifizierung/Interpretationen/AIS_31_Functionality_classes_evaluation_methodology_for_true_RNG_e.pdf?__blob=publicationFile&v=1
    https://www.bsi.bund.de/SharedDocs/Downloads/DE/BSI/Zertifizierung/Interpretationen/AIS_31_Functionality_classes_for_random_number_generators_e.pdf?__blob=publicationFile&v=1

    :param bitstream: A string of 0s and 1s representing the input random data.
    :type bitstream: str

    :return: True if the Test succeeded, False if it failed and None if there was an error.
    """
    N = 20000
    Tmax = 5000

    # Check if we have sufficient random data
    if len(bitstream) < N:
        perror("Not enough random data to perform this test.")
        return None

    # convert to int too speed things up
    bitstream = [int(x) for x in bitstream]

    # This is what is described in the paper but not what is implemented in the reference implementation of the BSI...
    #
    #    # Check the correlation for each shift
    #    for t in range(1, Tmax+1):
    #        T = sum([bitstream[j] ^ bitstream[j+t] for j in range(Tmax)])
    #        if not (2326 < T < 2674):
    #            print("Got T = {}, Expected : 2326 < T < 2674".format(T))
    #            return False
    #
    #        return True

    # calculate all shifts from 2500
    # old very inefficient code
    # m = 0
    # mi = 0
    # for t in range(1, Tmax+1):
    #    T = sum([bitstream[j] ^ bitstream[j + t] for j in range(Tmax)])
    #    shift = abs(T - (Tmax//2))
    #    # get the first t that produced the max shift
    #    if shift > m:
    #        mi = t
    #        m = shift

    # math trick to be able to use numpy.correlate
    p = N * 100
    # insanely fast !
    cor = numpy.correlate(bitstream[:Tmax], bitstream[1:Tmax * 2], "valid")[::-1]
    cor %= (p ** 2 - 1) // p  # correct values
    cor = list(numpy.abs(cor - Tmax // 2))
    m = max(cor)
    mi = cor.index(m) + 1

    # check the intervall on the second half of bits
    T = sum([bitstream[j] ^ bitstream[j + mi] for j in range(N // 2, N // 2 + Tmax)])
    if not (2326 < T < 2674):
        print("Got T = {}, Expected : 2326 < T < 2674".format(T))
        return False

    return True


def uniform_distribution_test(bitstream, k=2, n=100000, a=0.02):
    """
    BSI test T6.
    https://www.bsi.bund.de/SharedDocs/Downloads/DE/BSI/Zertifizierung/Interpretationen/AIS_31_Functionality_classes_evaluation_methodology_for_true_RNG_e.pdf?__blob=publicationFile&v=1
    https://www.bsi.bund.de/SharedDocs/Downloads/DE/BSI/Zertifizierung/Interpretationen/AIS_31_Functionality_classes_for_random_number_generators_e.pdf?__blob=publicationFile&v=1

    :param bitstream: A string of 0s and 1s representing the input random data.
    :type bitstream: str

    :paramutils k: Length of the vectors to be tested.
    :type k: int

    :param n: Length of the sequence to be tested.
    :type n: int

    :param a: Positive real number.
    :type a: float

    :return: True if the Test succeeded, False if it failed and None if there was an error.
    """

    # Check if we have sufficient random data
    if len(bitstream) < k * n:
        perror("Not enough random data to perform this test.")
        return None

    # Count the occurrences of the n k-bit numbers
    occurrences = [0] * 2 ** k
    for i in range(n):
        # construct a number from the bits
        start = i * k
        wi = int(bitstream[start:start + k], 2)
        occurrences[wi] += 1

    # special case k=1
    if k == 1:
        res = 0.5 - a <= occurrences[0] / n <= 0.5 + a
        if not res:
            print("Got Tx = {}, Expected : {} < Tx < {}".format(occurrences[0] / n, 0.5 - a, 0.5 + a))
        return res

    # k > 1 : Check if all T(x) pass the test
    for x in occurrences:
        Tx = x / n
        if not (1 / 2 ** k - a <= Tx <= 1 / 2 ** k + a):
            print("Got Tx = {}, Expected : {} < Tx < {}".format(Tx, 1 / 2 ** k - a, 1 / 2 ** k + a))
            return False

    return True


def homogeneity_test(bitstream, h=2, n=100000, rejection_limit=15.13):
    """
    BSI test T7.
    https://www.bsi.bund.de/SharedDocs/Downloads/DE/BSI/Zertifizierung/Interpretationen/AIS_31_Functionality_classes_evaluation_methodology_for_true_RNG_e.pdf?__blob=publicationFile&v=1
    https://www.bsi.bund.de/SharedDocs/Downloads/DE/BSI/Zertifizierung/Interpretationen/AIS_31_Functionality_classes_for_random_number_generators_e.pdf?__blob=publicationFile&v=1

    :param bitstream: A string of 0s and 1s representing the input random data.
    :type bitstream: str

    :param h: Length of the vectors to be tested.
    :type h: int
    
    :param n: Length of the sequence to be tested.
    :type n: int
    
    :param rejection_limit: The threshold determining if the test is passed or failed.
    :type rejection_limit: float

    :return: True if the Test succeeded, False if it failed and None if there was an error.
    """

    # Check if we have sufficient random data
    if len(bitstream) < h * n:
        perror("Not enough random data to perform this test.")
        return None

    # See the vectors composed of h bits (w11, w21, ..., wh1) as a grid of n lines, each line representing
    # a single element:

    # w11, ..., wh1
    # ...,
    # w1n, ..., whn
    # 1 <= i <= n

    # Count the number of 0s in the column i
    fi0 = [0] * h
    # Count the number of 1s in the column i (could be deduced from fi0, but more readable)
    fi1 = [0] * h
    for i in range(n):
        # get a line
        start = i * h
        element = bitstream[start:start + h]
        # iterate over each columns (could be done otherwise because h=2, but what if some day we need
        # another value for h ?)
        for col in range(h):
            v = element[col]
            if v == "1":
                fi1[col] += 1
            else:
                fi0[col] += 1

    # Calculate the probabilities to have a 1 or a 0
    P1 = sum(fi1) / (h * n)
    P0 = sum(fi0) / (h * n)

    # Calculate X2 (chi squared)
    T = 0
    for i in range(h):
        x0 = ((fi0[i] - n * P0) ** 2) / (n * P0)
        x1 = ((fi1[i] - n * P1) ** 2) / (n * P1)
        T += x0 + x1

    res = T <= rejection_limit
    if not res:
        print("Got T = {}, Expected : T <= {}".format(T, rejection_limit))
    return res


def entropy_test(bitstream, L=8, Q=2560, K=256000, rejection_limit=7.976):
    """
    BSI test T8.
    https://www.bsi.bund.de/SharedDocs/Downloads/DE/BSI/Zertifizierung/Interpretationen/AIS_31_Functionality_classes_evaluation_methodology_for_true_RNG_e.pdf?__blob=publicationFile&v=1
    https://www.bsi.bund.de/SharedDocs/Downloads/DE/BSI/Zertifizierung/Interpretationen/AIS_31_Functionality_classes_for_random_number_generators_e.pdf?__blob=publicationFile&v=1

    :param bitstream: A string of 0s and 1s representing the input random data.
    :type bitstream: str

    :param L: Length of the vectors to be tested.
    :type L: int

    :param Q: Test parameters.
    :type Q: int

    param K: Test parameters.
    :type K: int

    :param rejection_limit: The threshold determining if the test is passed or failed.
    :type rejection_limit: float

    :return: True if the Test succeeded, False if it failed and None if there was an error.
    """

    # Check if we have sufficient random data
    if len(bitstream) < (Q + K) * L:
        perror("Not enough random data to perform this test.")
        return None

    # define gi
    def g(i):
        import math
        s = sum([1 / k for k in range(1, i)])
        return s / math.log(2)

    # This holds the last index of each possible vector
    nearest = [0] * 2 ** L
    Fc = 0
    for i in range(Q + K):
        start = i * L
        Wi = int(bitstream[start:start + L], 2)
        Ai = i - nearest[Wi]
        nearest[Wi] = i
        # Compute Fc
        if i >= Q:
            Fc += g(Ai)
    Fc /= K

    res = Fc > rejection_limit
    if not res:
        print("Got Fc = {}, Expected : Fc > {}".format(Fc, rejection_limit))
    return res


# -------------
# Mappings
# -------------

TEST_FUNC = {
    BSI_TESTS.disjointness: disjointness_test,
    BSI_TESTS.monobit: monobit_test,
    BSI_TESTS.poker: poker_test,
    BSI_TESTS.runs: runs_test,
    BSI_TESTS.longrun: longrun_test,
    BSI_TESTS.autocorrelation: autocorrelation_test,
    BSI_TESTS.uniform_distribution: uniform_distribution_test,
    BSI_TESTS.homogeneity: homogeneity_test,
    BSI_TESTS.entropy: entropy_test
}

MIN_LENGTHS = {
    BSI_TESTS.disjointness: 2 ** 16 * 48,
    BSI_TESTS.monobit: 20000,
    BSI_TESTS.poker: 20000,
    BSI_TESTS.runs: 20000,
    BSI_TESTS.longrun: 20000,
    BSI_TESTS.autocorrelation: 20000,
    BSI_TESTS.uniform_distribution: 100000 * 2,
    BSI_TESTS.homogeneity: 100000 * 2,
    BSI_TESTS.entropy: (256000 + 2560) * 8
}


# -------------
# Procedures
# -------------

def logAndDump(bits, count, name):
    import binascii

    if count == 1:
        pwarning("First failure on {}".format(name))
        pwarning("It can happen with a very small probability...")

    if count > 1:
        perror("New failure on {}".format(name))
        perror("This should never happen on a good PRNG.")

    path = "dump/{}".format(time.time())
    with open(path, "wb") as f:
        raw = hex(int(bits, 2))[2:]
        if len(raw) % 2 != 0:
            raw = "0" + raw
        raw = binascii.unhexlify(raw)
        f.write(raw)

    print("Dump of the faulty data generated at : {}".format(path))


def procedure_A(getBits, window):
    """
    Applies the procedure A from the BSI.
    https://www.bsi.bund.de/SharedDocs/Downloads/DE/BSI/Zertifizierung/Interpretationen/AIS_31_Functionality_classes_for_random_number_generators_e.pdf?__blob=publicationFile&v=1

    :param getBits: Function that take the requested amount of bits as input parameter and returns them as string
        or None if the source can't provide that much.
    :type getBits: (int) -> str

    :return: True if the Test succeeded, False if it failed and None if there was an error.
    """

    window.resetInternalProgressBar(258)
    failCount = 0
    # step 1
    bits = getBits(2 ** 16 * 48)
    if bits is None:
        pwarning("Not enough bits to perform step 1 of procedure A.")
        return None
    if not disjointness_test(bits):
        failCount += 1
        logAndDump(bits, failCount, "disjointness test")

    window.incrementInternalProgressBar()
    # step 2 - 6
    chunk_size = 20000
    bits = getBits(chunk_size * 257)
    if bits is None:
        pwarning("Not enough bits to perform steps 2 to 6 of procedure A.")
        return None
    for i in range(257):
        # prendre bits par chunk de 20 000
        start = i * chunk_size
        chunk = bits[start:start + chunk_size]

        if not monobit_test(chunk):
            failCount += 1
            logAndDump(chunk, failCount, "monobit test")
        if not poker_test(chunk):
            failCount += 1
            logAndDump(chunk, failCount, "poker test")
        if not runs_test(chunk):
            failCount += 1
            logAndDump(chunk, failCount, "runs test")
        if not longrun_test(chunk):
            failCount += 1
            logAndDump(chunk, failCount, "long run test")
        if not autocorrelation_test(chunk):
            failCount += 1
            logAndDump(chunk, failCount, "autocorrelation test")

        # check on each turn so we don't have to wait in case of failure
        if failCount > 2:
            return False
        window.incrementInternalProgressBar()

    # All success (good RNG, probability of fail is approximately 0
    # if only 1 has failed retry with new data
    if failCount == 1:
        pwarning("Exactly one test has failed, apply the tests once more on a new data set to be sure.")
    return failCount == 0


def procedure_B(getBits, window):
    """
    Applies the procedure B from the BSI.
    https://www.bsi.bund.de/SharedDocs/Downloads/DE/BSI/Zertifizierung/Interpretationen/AIS_31_Functionality_classes_for_random_number_generators_e.pdf?__blob=publicationFile&v=1

    :param getBits: Function that take the requested amount of bits as input parameter and returns them as string
        or None if the source can't provide that much.
    :type getBits: (int) -> str

    :return: True if the Test succeeded, False if it failed and None if there was an error.
    """
    window.resetInternalProgressBar(5)
    failCount = 0

    bits = getBits(100000)
    if bits is None:
        pwarning("Not enough bits to perform step 1 of procedure B.")
        return None
    if not uniform_distribution_test(bits, 1, 100000, 0.025):
        failCount += 1
        logAndDump(bits, failCount, "uniform distribution test with params 1, 100000, 0.025")
    window.incrementInternalProgressBar()

    bits = getBits(100000 * 2)
    if bits is None:
        pwarning("Not enough bits to perform step 2 of procedure B.")
        return None
    if not uniform_distribution_test(bits, 2, 100000, 0.02):
        failCount += 1
        logAndDump(bits, failCount, "uniform distribution test with params 2, 100000, 0.02")
    window.incrementInternalProgressBar()

    bits = getBits(100000 * 3)
    if bits is None:
        pwarning("Not enough bits to perform step 3 of procedure B.")
        return None
    if not homogeneity_test(bits, 3):
        failCount += 1
        logAndDump(bits, failCount, "homogeneity test with param 3")
    window.incrementInternalProgressBar()

    bits = getBits(100000 * 4)
    if bits is None:
        pwarning("Not enough bits to perform step 4 of procedure B.")
        return None
    if not homogeneity_test(bits, 4):
        failCount += 1
        logAndDump(bits, failCount, "homogeneity test with param 4")
    window.incrementInternalProgressBar()

    bits = getBits((256000 + 2560) * 8)
    if bits is None:
        pwarning("Not enough bits to perform step 5 of procedure B.")
        return None
    if not entropy_test(bits):
        failCount += 1
        logAndDump(bits, failCount, "entropy test")
    window.incrementInternalProgressBar()

    # All success (good RNG, probability of fail is approximately 0
    # if only 1 has failed retry with new data
    if failCount == 1:
        pwarning("Exactly one test has failed, apply the tests once more on a new data set to be sure.")
    return failCount == 0
