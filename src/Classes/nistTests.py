# Author : Florian Picca <florian.picca@oppida.fr>
# Date : May 2020

from ..utils import *
import math
import mpmath
from scipy.special import gammaincc


# -------------------
# Individual tests
# -------------------

def monobit(bits):
    """
    SP80022r1A p24-p25 - 2.1.4
    https://nvlpubs.nist.gov/nistpubs/legacy/sp/nistspecialpublication800-22r1a.pdf

    :param bits: A string of 0s and 1s representing the input random data.
    :type bits: str

    :return: True if the Test succeeded, False if it failed and None if there was an error.
    """
    n = len(bits)
    # Minimum recommended length
    if n < 100:
        perror("Not enough random data to perform this test.")
        return None
    # compute the sum of the converted bits
    # 0 -> -1
    # 1 -> 1
    S = abs(sum([2 * int(e) - 1 for e in bits]))

    Pv = math.erfc(S / math.sqrt(n) / math.sqrt(2))
    return Pv >= 0.01


def frequencyBlock(bits, M=None):
    """
    SP80022r1A p26-p27 - 2.2.4
    https://nvlpubs.nist.gov/nistpubs/legacy/sp/nistspecialpublication800-22r1a.pdf

    :param bits: A string of 0s and 1s representing the input random data.
    :type bits: str

    :param M: The length of each block.
    :type M: int

    :return: True if the Test succeeded, False if it failed and None if there was an error.
    """
    n = len(bits)
    if M is None:
        # something that satisfies the conditions by default
        M = n // 10
    N = n // M
    # Minimum recommended length
    # last 2 conditions are redondant
    if n < 100 or M < 20 or M <= n // 100 or N >= 100:
        perror("Conditions not satisfied to run this test.")
        return None
    # Compute proportion of 1s in each chunk
    pi = []
    for i in range(N):
        chunk = bits[M * i:(M * i) + M]
        ones = chunk.count("1")
        pi.append(ones / M)
    # Compute chi squared
    chi = 4 * M
    chi *= sum([(x - 0.5) ** 2 for x in pi])
    Pv = gammaincc(N / 2, chi / 2)
    return Pv >= 0.01


def runs(bits):
    """
    SP80022r1A p28-p29 - 2.3.4
    https://nvlpubs.nist.gov/nistpubs/legacy/sp/nistspecialpublication800-22r1a.pdf

    :param bits: A string of 0s and 1s representing the input random data.
    :type bits: str

    :return: True if the Test succeeded, False if it failed and None if there was an error.
    """
    n = len(bits)
    # Minimum recommended length
    if n < 100:
        perror("Not enough random data to perform this test.")
        return None
    pi = bits.count("1") / n
    tau = 2 / math.sqrt(n)
    # Check if this test should be run, a failure should mean that this test should have failed on the monobit test
    # but experimenting showed that it's not the case
    if abs(pi - 0.5) >= tau:
        return False
    # Add the last one here, doesn't matter anyway
    r = "1"
    for i in range(len(bits) - 1):
        if bits[i] == bits[i + 1]:
            r += "0"
        else:
            r += "1"
    V = r.count("1")
    x = pi * (1 - pi)
    Pv = math.erfc(abs(V - 2 * n * x) / (2 * math.sqrt(2 * n) * x))
    return Pv >= 0.01


def longestRun(bits):
    """
    SP80022r1A p30-p31 - 2.4.4
    https://nvlpubs.nist.gov/nistpubs/legacy/sp/nistspecialpublication800-22r1a.pdf

    :param bits: A string of 0s and 1s representing the input random data.
    :type bits: str

    :return: True if the Test succeeded, False if it failed and None if there was an error.
    """
    n = len(bits)
    # Minimum recommended length
    if n < 128:
        perror("Not enough random data to perform this test.")
        return None

    # K is 1 more than in the paper for convenience
    if n < 6272:
        M = 8
        K = 4
        pi = [0.2148, 0.3672, 0.2305, 0.1875]
    elif n < 750000:
        M = 128
        K = 6
        pi = [0.1174, 0.2430, 0.2493, 0.1752, 0.1027, 0.1124]
    else:
        M = 10000
        K = 7
        pi = [0.0882, 0.2092, 0.2483, 0.1933, 0.1208, 0.0675, 0.0727]

    V = [0] * K
    N = n // M
    # For each block, compute the longest run
    for i in range(N):
        chunk = bits[M * i:(M * i) + M]
        longest = 1
        while chunk.count("1" * longest) > 0:
            longest += 1
        longest -= 1
        # Compute the index of the cell in V
        if M == 8:
            index = longest - 1
        elif M == 128:
            index = longest - 4
        else:
            index = longest - 10
        # handle cases when run is in the first cell or last cell
        if index < 0:
            index = 0
        if index >= K:
            index = K - 1
        # increment the right cell of V
        V[index] += 1
    # compute chi squared
    chi = sum([(V[i] - N * pi[i]) ** 2 / (N * pi[i]) for i in range(K)])
    K -= 1
    Pv = gammaincc(K / 2, chi / 2)
    return Pv >= 0.01


def binaryMatrixRank(bits):
    """
    SP80022r1A p32-p34 - 2.5.4
    https://nvlpubs.nist.gov/nistpubs/legacy/sp/nistspecialpublication800-22r1a.pdf

    :param bits: A string of 0s and 1s representing the input random data.
    :type bits: str

    :return: True if the Test succeeded, False if it failed and None if there was an error.
    """
    n = len(bits)
    M = Q = 32

    # Minimum recommended length
    if n < 38 * M * Q:
        perror("Not enough random data to perform this test.")
        return None

    N = n // (M * Q)
    # count number of matrices with rank = M, M-1 and the rest
    Fm = Fm1 = Fmr = 0
    # construct the matrices and compute rank
    for i in range(N):
        start = i * M * Q
        m = [[int(e) for e in bits[start + j:start + j + Q]] for j in range(0, M * Q, Q)]
        r = rankMat(m)
        if r == M:
            Fm += 1
        elif r == M - 1:
            Fm1 += 1
        else:
            Fmr += 1
    # chi2
    T = (Fm - 0.2888 * N) ** 2 / (0.2888 * N)
    T += (Fm1 - 0.5776 * N) ** 2 / (0.5776 * N)
    T += (Fmr - 0.1336 * N) ** 2 / (0.1336 * N)
    Pv = gammaincc(1, T / 2)
    return Pv >= 0.01


def nonOverlappingTemplate(bits, template="111110100", N=8):
    """
    SP80022r1A p36-p37 - 2.7.4
    https://nvlpubs.nist.gov/nistpubs/legacy/sp/nistspecialpublication800-22r1a.pdf

    :param bits: A string of 0s and 1s representing the input random data.
    :type bits: str

    :param template: A string of 0s and 1s representing the template to search for.
    :type template: str

    :param N: The number of blocks to make.
    :type N: int

    :return: True if the Test succeeded, False if it failed and None if there was an error.
    """
    # template must be aperiodic
    if not isAperiodic(template):
        perror("The template must be aperiodic.")
        return None

    n = len(bits)
    m = len(template)
    M = n // N
    # Minimum recommended length
    if M < m or N > 100 or m > 10 or m < 9:
        perror("Conditions not satisfied to run this test.")
        return None

    W = []
    # Count the non-overlapping template occurrences in each block
    for i in range(N):
        chunk = bits[M * i:(M * i) + M]
        W.append(chunk.count(template))

    mu = (M - m + 1) / 2 ** m
    sigma2 = M * (1 / 2 ** m - (2 * m - 1) / 2 ** (2 * m))
    chi = sum([(w - mu) ** 2 / sigma2 for w in W])
    Pv = gammaincc(N / 2, chi / 2)
    return Pv >= 0.01


def overlappingTemplate(bits, template="111111111", N=968):
    """
    SP80022r1A p39-p40 - 2.8.4
    https://nvlpubs.nist.gov/nistpubs/legacy/sp/nistspecialpublication800-22r1a.pdf

    :param bits: A string of 0s and 1s representing the input random data.
    :type bits: str

    :param template: A string of 0s and 1s representing the template to search for.
    :type template: str

    :param N: The number of blocks to make.
    :type N: int

    :return: True if the Test succeeded, False if it failed and None if there was an error.
    """
    # template must be periodic
    if isAperiodic(template):
        perror("The template must be periodic.")
        return None

    n = len(bits)
    m = len(template)
    M = n // N
    # Minimum recommended length
    if n < 10 ** 6 or m > 10 or m < 9:
        perror("Conditions not satisfied to run this test.")
        return None

    V = [0] * 6
    # Count the non-overlapping template occurrences in each block
    for i in range(N):
        chunk = bits[M * i:(M * i) + M]
        count = start = 0
        while True:
            start = chunk.find(template, start) + 1
            if start > 0:
                count += 1
            else:
                break
        if count >= 5:
            V[5] += 1
        else:
            V[count] += 1
    lbd = (M - m + 1) / 2 ** m
    mu = lbd / 2

    def get_prob(u, x):
        import numpy
        from scipy.special import hyp1f1
        out = 1.0 * numpy.exp(-x)
        if u != 0:
            out = 1.0 * x * numpy.exp(2 * -x) * (2 ** -u) * hyp1f1(u + 1, 2, x)
        return out

    pi = [get_prob(i, mu) for i in range(5)]
    diff = sum(pi)
    pi.append(1.0 - diff)
    chi = sum([(V[i] - N * pi[i]) ** 2 / (N * pi[i]) for i in range(6)])
    Pv = gammaincc(5 / 2, chi / 2)
    return Pv >= 0.01


def linearComplexity(bits, M=1000):
    """
    SP80022r1A p46-p48 - 2.10.4
    https://nvlpubs.nist.gov/nistpubs/legacy/sp/nistspecialpublication800-22r1a.pdf

    :param bits: A string of 0s and 1s representing the input random data.
    :type bits: str

    :param M: The length of each block.
    :type M: int

    :return: True if the Test succeeded, False if it failed and None if there was an error.
    """
    n = len(bits)
    N = n // M

    if n < 10 ** 6 or M < 500 or M > 5000 or N < 200:
        perror("Conditions not satisfied to run this test.")
        return None

    bits = [int(e) for e in bits]
    # constants
    pi = [0.010417, 0.03125, 0.125, 0.5, 0.25, 0.0625, 0.020833]
    # Theoretical value
    mu = M / 2 + ((9 + pow(-1, M + 1)) / 36) - (M / 3 + 2 / 9) / pow(2, M)
    V = [0] * 7
    for i in range(N):
        chunk = bits[i * M:i * M + M]
        L = berlekamp_massey(chunk)
        T = pow(-1, M) * (L - mu) + 2 / 9
        if T <= -2.5:
            V[0] += 1
        elif -2.5 < T <= -1.5:
            V[1] += 1
        elif -1.5 < T <= -0.5:
            V[2] += 1
        elif -0.5 < T <= 0.5:
            V[3] += 1
        elif 0.5 < T <= 1.5:
            V[4] += 1
        elif 1.5 < T <= 2.5:
            V[5] += 1
        else:
            V[6] += 1
    # chi2
    X = 0
    for i in range(7):
        X += (V[i] - N * pi[i]) ** 2 / (N * pi[i])
    # Might need to use the Decimal module for this test, values are not exactly the ones from the paper
    # but close enough
    Pv = gammaincc(3, X / 2)
    return Pv >= 0.01


def cumulativeSums(bits):
    """
    SP80022r1A p53-p54 - 2.13.4
    mode forward only
    https://nvlpubs.nist.gov/nistpubs/legacy/sp/nistspecialpublication800-22r1a.pdf

    :param bits: A string of 0s and 1s representing the input random data.
    :type bits: str

    :return: True if the Test succeeded, False if it failed and None if there was an error.
    """
    n = len(bits)
    # Minimum recommended length
    if n < 100:
        perror("Not enough random data to perform this test.")
        return None

    X = [2 * int(e) - 1 for e in bits]
    S = [0] * n

    S[0] = X[0]
    for i in range(1, n):
        S[i] = S[i - 1] + X[i]

    z = max([abs(x) for x in S])

    Pv = 1
    inf = int((-n / z + 1) / 4)
    sup = int((n / z - 1) / 4) + 1
    Pv -= sum([mpmath.ncdf((4 * k + 1) * z / math.sqrt(n)) - mpmath.ncdf((4 * k - 1) * z / math.sqrt(n)) for k in
               range(inf, sup)])
    inf = int((-n / z - 3) / 4)
    sup = int((n / z - 1) / 4) + 1
    Pv += sum([mpmath.ncdf((4 * k + 3) * z / math.sqrt(n)) - mpmath.ncdf((4 * k + 1) * z / math.sqrt(n)) for k in
               range(inf, sup)])
    return Pv >= 0.01


def serialTest(bits, m=3):
    """
    SP80022r1A p48-p51 - 2.11.4
    https://nvlpubs.nist.gov/nistpubs/legacy/sp/nistspecialpublication800-22r1a.pdf

    :param bits: A string of 0s and 1s representing the input random data.
    :type bits: str

    :param m: The length of each block.
    :type m: int

    :return: True if the Test succeeded, False if it failed and None if there was an error.
    """
    n = len(bits)

    if m >= math.log2(n) - 2:
        perror("Conditions not satisfied to run this test.")
        return None

    T = []
    for M in range(m, m - 3, -1):
        if M <= 0:
            T.append(0)
            continue
        e = bits + bits[:M - 1]
        Y2 = 2 ** M / n
        s = 0
        # Count the non-overlapping template occurrences in each block
        for t in range(2 ** M):
            template = bin(t)[2:].zfill(M)
            count = start = 0
            while True:
                start = e.find(template, start) + 1
                if start > 0:
                    count += 1
                else:
                    break
            s += count ** 2
        Y2 *= s
        Y2 -= n
        T.append(Y2)
    D1 = T[0] - T[1]
    D2 = T[0] - 2 * T[1] + T[2]
    Pv1 = gammaincc(2 ** (m - 2), D1 / 2)
    Pv2 = gammaincc(2 ** (m - 3), D2 / 2)
    return Pv1 > 0.01 and Pv2 > 0.01


def approximateEntropie(bits, m=3):
    """
    SP80022r1A p51-p54 - 2.12.4
    https://nvlpubs.nist.gov/nistpubs/legacy/sp/nistspecialpublication800-22r1a.pdf

    :param bits: A string of 0s and 1s representing the input random data.
    :type bits: str

    :param m: The length of each block.
    :type m: int

    :return: True if the Test succeeded, False if it failed and None if there was an error.
    """
    n = len(bits)

    if m >= math.log2(n) - 5:
        perror("Conditions not satisfied to run this test.")
        return None

    T = []
    for M in range(m, m + 2):
        e = bits + bits[:M - 1]
        Phi = 0
        # Count the non-overlapping template occurrences in each block
        for t in range(2 ** M):
            template = bin(t)[2:].zfill(M)
            count = start = 0
            while True:
                start = e.find(template, start) + 1
                if start > 0:
                    count += 1
                else:
                    break
            pi = count / n
            if pi > 0:
                # avoid domain error as log(0) is not defined
                Phi += pi * math.log(pi)
        T.append(Phi)
    chi = 2 * n * (math.log(2) - (T[0] - T[1]))
    Pv = gammaincc(2 ** (m - 1), chi / 2)
    return Pv >= 0.01


def randomExcursion(bits):
    """
    SP80022r1A p56-p58 - 2.14.4
    https://nvlpubs.nist.gov/nistpubs/legacy/sp/nistspecialpublication800-22r1a.pdf

    :param bits: A string of 0s and 1s representing the input random data.
    :type bits: str

    :return: True if the Test succeeded, False if it failed and None if there was an error.
    """
    n = len(bits)

    # Minimum recommended length
    if n < 10 ** 6:
        perror("Not enough random data to perform this test.")
        return None

    X = [2 * int(e) - 1 for e in bits]

    def pi(k, x):
        if k == 0:
            return 1 - 1 / (2 * abs(x))
        elif k >= 5:
            return 1 / (2 * abs(x)) * pow(1 - (1 / (2 * abs(x))), 4)
        return 1 / (4 * x ** 2) * pow(1 - (1 / (2 * abs(x))), k - 1)

    S = X[0]
    cycles = []
    cy = [S]
    for i in range(1, n):
        S += X[i]
        # cycles
        if S == 0:
            cycles.append(cy)
            cy = []
            # skip default instruction
            continue
        cy.append(S)
    cycles.append(cy)
    J = len(cycles)

    # Minimum number of cycles for the stats to be correct
    if J < 500:
        perror("Not enough cycles to continue this test.")
        return None

    Xi = []
    for x in range(-4, 5):
        if x == 0:
            # skip 0
            continue
        L = [0] * 6
        for e in cycles:
            ctr = e.count(x)
            if ctr > 5:
                ctr = 5
            L[ctr] += 1
        # compute Xi
        T = sum([(L[i] - J * pi(i, x)) ** 2 / (J * pi(i, x)) for i in range(6)])
        Xi.append(T)
    Pv = [gammaincc(5 / 2, x / 2) for x in Xi]
    res = [x > 0.01 for x in Pv]
    f = res.count(False)
    if f > 0:
        pwarning(f"Failed tests: {f}/8")
    return all(res)


def randomExcursionVariant(bits):
    """
    SP80022r1A p60-p63 - 2.15.4
    https://nvlpubs.nist.gov/nistpubs/legacy/sp/nistspecialpublication800-22r1a.pdf

    :param bits: A string of 0s and 1s representing the input random data.
    :type bits: str

    :return: True if the Test succeeded, False if it failed and None if there was an error.
    """
    n = len(bits)

    # Minimum recommended length
    if n < 10 ** 6:
        perror("Not enough random data to perform this test.")
        return None

    X = [2 * int(e) - 1 for e in bits]

    def getIndex(x):
        index = x + 9
        if index >= 9:
            index -= 1
        return index

    S = 0
    eps = [0] * 18
    J = 1
    for i in range(n):
        S += X[i]
        if S != 0 and -9 <= S <= 9:
            eps[getIndex(S)] += 1
        # cycles
        if S == 0:
            J += 1
    Pv = [mpmath.erfc(abs(eps[getIndex(x)] - J) / math.sqrt(2 * J * (4 * abs(x) - 2))) for x in range(-9, 10) if x != 0]
    res = [x > 0.01 for x in Pv]
    f = res.count(False)
    if f > 0:
        pwarning(f"Failed tests: {f}/18")
    return all(res)


def maurer(bits):
    """
    SP80022r1A p42-p45 - 2.9.4
    https://nvlpubs.nist.gov/nistpubs/legacy/sp/nistspecialpublication800-22r1a.pdf

    :param bits: A string of 0s and 1s representing the input random data.
    :type bits: str

    :return: True if the Test succeeded, False if it failed and None if there was an error.
    """
    n = len(bits)

    # Minimum recommended length
    if n < 387840:
        perror("Not enough random data to perform this test.")
        return None
    if n < 904960:
        L = 6
        eV = 5.2177052
        variance = 2.954
    elif n < 2068480:
        L = 7
        eV = 6.1962507
        variance = 3.125
    elif n < 4654080:
        L = 8
        eV = 7.1836656
        variance = 3.238
    elif n < 10342400:
        L = 9
        eV = 8.1764248
        variance = 3.311
    elif n < 22753280:
        L = 10
        eV = 9.1723243
        variance = 3.356
    elif n < 49643520:
        L = 11
        eV = 10.170032
        variance = 3.384
    elif n < 107560960:
        L = 12
        eV = 11.168765
        variance = 3.401
    elif n < 231669760:
        L = 13
        eV = 12.168070
        variance = 3.410
    elif n < 496435200:
        L = 14
        eV = 13.167693
        variance = 3.416
    elif n < 1059061760:
        L = 15
        eV = 14.167488
        variance = 3.419
    else:
        L = 16
        eV = 15.167379
        variance = 3.421
    Q = 10 * 2 ** L
    K = n // L - Q

    # Init table
    T = [0] * 2 ** L
    for i in range(Q):
        b = bits[i * L:i * L + L]
        v = int(b, 2)
        # Store block index, starting at 1
        T[v] = i + 1

    # compute sum
    S = 0
    for i in range(Q, Q + K):
        b = bits[i * L:i * L + L]
        v = int(b, 2)
        old = T[v]
        T[v] = i + 1
        S += math.log2(i + 1 - old)

    Fn = S / K
    c = 0.7 - (0.8 / L) + (4 + (32 / L)) * (K ** (-3 / L) / 15)
    o = c * math.sqrt(variance / K)
    Pv = mpmath.erfc(abs(Fn - eV) / math.sqrt(2 * o))
    return Pv >= 0.01


def spectral(bits):
    """
    SP80022r1A p34-p36 - 2.6.4
    https://nvlpubs.nist.gov/nistpubs/legacy/sp/nistspecialpublication800-22r1a.pdf

    :param bits: A string of 0s and 1s representing the input random data.
    :type bits: str

    :return: True if the Test succeeded, False if it failed and None if there was an error.
    """
    import numpy
    n = len(bits)

    # Minimum recommended length
    if n < 1000:
        perror("Not enough random data to perform this test.")
        return None

    X = [2 * int(e) - 1 for e in bits]

    S = numpy.fft.fft(X)
    M = numpy.abs(S[:n // 2])
    T = math.sqrt(math.log(1 / 0.05) * n)
    N0 = 0.95 * n / 2
    N1 = len([x for x in M if x < T])
    d = (N1 - N0) / math.sqrt(n * 0.95 * 0.05 / 4)
    Pv = mpmath.erfc(abs(d) / math.sqrt(2))
    return Pv >= 0.01


# -------------
# Mappings
# -------------

TEST_FUNC = {
    NIST_TESTS.monobit: monobit,
    NIST_TESTS.frequency_block: frequencyBlock,
    NIST_TESTS.runs: runs,
    NIST_TESTS.longestrun: longestRun,
    NIST_TESTS.non_overlapping_template: nonOverlappingTemplate,
    NIST_TESTS.cusum: cumulativeSums,
    NIST_TESTS.binary_matrix_rank: binaryMatrixRank,
    NIST_TESTS.linear_complexity: linearComplexity,
    NIST_TESTS.overlapping_template: overlappingTemplate,
    NIST_TESTS.serial: serialTest,
    NIST_TESTS.approximate_entropy: approximateEntropie,
    NIST_TESTS.random_excursion: randomExcursion,
    NIST_TESTS.random_excursion_variant: randomExcursionVariant,
    NIST_TESTS.maurer: maurer,
    NIST_TESTS.spectral: spectral,
}

MIN_LENGTHS = {
    NIST_TESTS.monobit: 100,
    NIST_TESTS.frequency_block: 100,
    NIST_TESTS.runs: 100,
    NIST_TESTS.longestrun: 128,
    NIST_TESTS.non_overlapping_template: 100,
    NIST_TESTS.cusum: 100,
    NIST_TESTS.binary_matrix_rank: 38912,
    NIST_TESTS.linear_complexity: 1000000,
    NIST_TESTS.overlapping_template: 1000000,
    NIST_TESTS.serial: 100,
    NIST_TESTS.approximate_entropy: 100,
    NIST_TESTS.random_excursion: 1000000,
    NIST_TESTS.random_excursion_variant: 1000000,
    NIST_TESTS.maurer: 387840,
    NIST_TESTS.spectral: 1000,
}

# -------------
# Procedures
# -------------

# NIST does not define procedures, maybe just run all the tests ?
