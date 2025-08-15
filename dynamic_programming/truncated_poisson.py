import numpy as np


class TruncatedPoisson:
    """
    A class that allows us to compute and cache the PMF and expectations of a truncated poisson distribution.
    
    Initialize a new instance for different lambda parameters. The class will provide utility functions to compute
    cached PMF values at any truncation val (up to a max limit) along with the expected values at those truncation values.
    """

    def __init__(self, lambda_val: float, max_trunc_val: int) -> None:
        self._lambda = lambda_val
        self._max_trunc_val = max_trunc_val

        # initialize the containers
        self._pmfs, self._cdfs = self._compute_pmfs_cdfs()
        self._expectations = self._compute_expectations()
    
    def pmf(self, trunc_val: int, num: int) -> float:
        assert 0 <= num <= trunc_val

        if num < trunc_val:
            return self._pmfs[num]
        
        if trunc_val == 0:
            return 1

        return 1 - self._cdfs[num - 1]
    
    def expectation(self, trunc_val: int) -> float:
        return self._expectations[trunc_val]
    
    def _compute_pmfs_cdfs(self) -> np.ndarray:
        """
        Helper function to one time compute the PMFs and CDFs for the poisson RV up to max_trunc_val
        """

        # compute the pmfs for the standard poisson up to max_trunc_val
        curr_exponent = 1
        curr_factorial = 1
        pmfs = [1]
        for num in range(1, self._max_trunc_val + 1):
            curr_exponent *= self._lambda
            curr_factorial *= num
            pmfs.append(curr_exponent / curr_factorial)
        pmfs = np.array(pmfs) * np.exp(-self._lambda)

        # compute the cdfs
        cdfs = np.cumsum(pmfs)
        
        return pmfs, cdfs

    def _compute_expectations(self) -> np.ndarray:
        """
        Helper funtion to one time compute the expectations given the pmfs and cdfs
        """

        # rolling sum is the expectation offset by one posistion
        expectations = np.cumsum(1 - self._cdfs)

        # concat a zero to the beginning
        return np.concat([[0], expectations])
