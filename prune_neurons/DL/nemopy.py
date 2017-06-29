from scipy.stats import friedmanchisquare, rankdata, norm
from scipy.special import gammaln
import numpy as np


# consistent with https://cran.r-project.org/web/packages/PMCMR/vignettes/PMCMR.pdf p. 17


class NemenyiTestPostHoc():

    def __init__(self, data):
        self._noOfGroups = data.shape[0]
        self._noOfSamples = data.shape[1]
        self._data = data

    def do(self):
        dataAsRanks = np.full(self._data.shape, np.nan)
        for i in range(self._noOfSamples):
            dataAsRanks[:, i] = rankdata(self._data[:, i])
        meansOfRanksOfDependentSamples = np.mean(dataAsRanks, 1)
        qValues = self._compareStatisticsOfAllPairs(meansOfRanksOfDependentSamples)
        pValues = self._calculatePValues(qValues)

        return meansOfRanksOfDependentSamples, pValues

    def _compareStatisticsOfAllPairs(self, meansOfRanks):
        noOfMeansOfRanks = len(meansOfRanks)
        compareResults = np.zeros((noOfMeansOfRanks-1, noOfMeansOfRanks))
        for i in range(noOfMeansOfRanks-1):
            for j in range(i+1, noOfMeansOfRanks):
                compareResults[i][j] = self._compareStatisticsOfSinglePair((meansOfRanks[i], meansOfRanks[j]))
        return compareResults

    def _compareStatisticsOfSinglePair(self, meansOfRanksPair):
        diff = abs(meansOfRanksPair[0] - meansOfRanksPair[1])
        qval = diff / np.sqrt(self._noOfGroups * (self._noOfGroups + 1) / (6 * self._noOfSamples))
        return qval * np.sqrt(2)

    def _calculatePValues(self, qValues):
        for qRow in qValues:
            for i in range(len(qRow)):
                qRow[i] = self._ptukey(qRow[i], 1, self._noOfGroups, np.inf)
        return 1 - qValues

    def _wprob(self, w, rr, cc):
        nleg = 12
        ihalf = 6

        C1 = -30
        C2 = -50
        C3 = 60
        M_1_SQRT_2PI = 1 / np.sqrt(2 * np.pi)
        bb = 8
        wlar = 3
        wincr1 = 2
        wincr2 = 3
        xleg = [
            0.981560634246719250690549090149,
            0.904117256370474856678465866119,
            0.769902674194304687036893833213,
            0.587317954286617447296702418941,
            0.367831498998180193752691536644,
            0.125233408511468915472441369464
        ]
        aleg = [
            0.047175336386511827194615961485,
            0.106939325995318430960254718194,
            0.160078328543346226334652529543,
            0.203167426723065921749064455810,
            0.233492536538354808760849898925,
            0.249147045813402785000562436043
        ]

        qsqz = w * 0.5

        if qsqz >= bb:
            return 1.0

        # find (f(w/2) - 1) ^ cc
        # (first term in integral of hartley's form).

        pr_w = 2 * norm.cdf(qsqz) - 1
        if pr_w >= np.exp(C2 / cc):
            pr_w = pr_w ** cc
        else:
            pr_w = 0.0

        # if w is large then the second component of the
        # integral is small, so fewer intervals are needed.

        wincr = wincr1 if w > wlar else wincr2

        # find the integral of second term of hartley's form
        # for the integral of the range for equal-length
        # intervals using legendre quadrature.  limits of
        # integration are from (w/2, 8).  two or three
        # equal-length intervals are used.

        # blb and bub are lower and upper limits of integration.

        blb = qsqz
        binc = (bb - qsqz) / wincr
        bub = blb + binc
        einsum = 0.0

        # integrate over each interval

        cc1 = cc - 1.0
        for wi in range(1, wincr + 1):
            elsum = 0.0
            a = 0.5 * (bub + blb)

            # legendre quadrature with order = nleg

            b = 0.5 * (bub - blb)

            for jj in range(1, nleg + 1):
                if (ihalf < jj):
                    j = (nleg - jj) + 1
                    xx = xleg[j-1]
                else:
                    j = jj
                    xx = -xleg[j-1]
                c = b * xx
                ac = a + c

                # if exp(-qexpo/2) < 9e-14
                # then doesn't contribute to integral

                qexpo = ac * ac
                if qexpo > C3:
                    break

                pplus = 2 * norm.cdf(ac)
                pminus = 2 * norm.cdf(ac, w)

                # if rinsum ^ (cc-1) < 9e-14, */
                # then doesn't contribute to integral */

                rinsum = (pplus * 0.5) - (pminus * 0.5)
                if (rinsum >= np.exp(C1 / cc1)):
                    rinsum = (aleg[j-1] * np.exp(-(0.5 * qexpo))) * (rinsum ** cc1)
                    elsum += rinsum

            elsum *= (((2.0 * b) * cc) * M_1_SQRT_2PI)
            einsum += elsum
            blb = bub
            bub += binc

        # if pr_w ^ rr < 9e-14, then return 0
        pr_w += einsum
        if pr_w <= np.exp(C1 / rr):
            return 0

        pr_w = pr_w ** rr
        if (pr_w >= 1):
            return 1
        return pr_w

    def _ptukey(self, q, rr, cc, df):

        M_LN2 = 0.69314718055994530942

        nlegq = 16
        ihalfq = 8

        eps1 = -30.0
        eps2 = 1.0e-14
        dhaf = 100.0
        dquar = 800.0
        deigh = 5000.0
        dlarg = 25000.0
        ulen1 = 1.0
        ulen2 = 0.5
        ulen3 = 0.25
        ulen4 = 0.125
        xlegq = [
            0.989400934991649932596154173450,
            0.944575023073232576077988415535,
            0.865631202387831743880467897712,
            0.755404408355003033895101194847,
            0.617876244402643748446671764049,
            0.458016777657227386342419442984,
            0.281603550779258913230460501460,
            0.950125098376374401853193354250e-1
        ]
        alegq = [
            0.271524594117540948517805724560e-1,
            0.622535239386478928628438369944e-1,
            0.951585116824927848099251076022e-1,
            0.124628971255533872052476282192,
            0.149595988816576732081501730547,
            0.169156519395002538189312079030,
            0.182603415044923588866763667969,
            0.189450610455068496285396723208
        ]

        if q <= 0:
            return 0

        if (df < 2) or (rr < 1) or (cc < 2):
            return float('nan')

        if np.isfinite(q) is False:
            return 1

        if df > dlarg:
            return self._wprob(q, rr, cc)

        # in fact we don't need the code below and majority of variables:

        # calculate leading constant

        f2 = df * 0.5
        f2lf = ((f2 * np.log(df)) - (df * M_LN2)) - gammaln(f2)
        f21 = f2 - 1.0

        # integral is divided into unit, half-unit, quarter-unit, or
        # eighth-unit length intervals depending on the value of the
        # degrees of freedom.

        ff4 = df * 0.25
        if df <= dhaf:
            ulen = ulen1
        elif df <= dquar:
            ulen = ulen2
        elif df <= deigh:
            ulen = ulen3
        else:
            ulen = ulen4

        f2lf += np.log(ulen)

        ans = 0.0

        for i in range(1, 51):
            otsum = 0.0

            # legendre quadrature with order = nlegq
            # nodes (stored in xlegq) are symmetric around zero.

            twa1 = (2*i - 1) * ulen

            for jj in range(1, nlegq + 1):
                if (ihalfq < jj):
                    j = jj - ihalfq - 1
                    t1 = (f2lf + (f21 * np.log(twa1 + (xlegq[j] * ulen)))) - (((xlegq[j] * ulen) + twa1) * ff4)
                else:
                    j = jj - 1
                    t1 = (f2lf + (f21 * np.log(twa1 - (xlegq[j] * ulen)))) + (((xlegq[j] * ulen) - twa1) * ff4)

                # if exp(t1) < 9e-14, then doesn't contribute to integral
                if t1 >= eps1:
                    if ihalfq < jj:
                        qsqz = q * np.sqrt(((xlegq[j] * ulen) + twa1) * 0.5)
                    else:
                        qsqz = q * np.sqrt(((-(xlegq[j] * ulen)) + twa1) * 0.5)

                    wprb = self._wprob(qsqz, rr, cc)
                    rotsum = (wprb * alegq[j]) * np.exp(t1)
                    otsum += rotsum

            # if integral for interval i < 1e-14, then stop.
            # However, in order to avoid small area under left tail,
            # at least  1 / ulen  intervals are calculated.

            if (i * ulen >= 1.0) and (otsum <= eps2):
                break

            ans += otsum

        return min(1, ans)
