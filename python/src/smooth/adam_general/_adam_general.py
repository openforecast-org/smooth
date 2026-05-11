"""
Python wrapper for C++ adamCore class methods.

This module provides backward-compatible wrappers that:
1. Maintain the old function-based API while using the new adamCore class
2. Convert C++ FitResult/ForecastResult objects to dictionaries
3. Handle type conversions between Python and C++
"""

import numpy as np

from smooth.adam_general import _adamCore


def adam_fitter(
    matrixVt,
    matrixWt,
    matrixF,
    vectorG,
    lags,
    indexLookupTable,
    profilesRecent,
    E,
    T,
    S,
    nNonSeasonal,
    nSeasonal,
    nArima,
    nXreg,
    constant,
    vectorYt,
    vectorOt,
    backcast,
    nIterations,
    refineHead,
    adamETS,
):
    """
    Wrapper for adamCore.fit() method.

    Converts function-style call to adamCore class method call
    and returns dict for backward compatibility.
    """
    # Convert types to ensure C++ compatibility
    lags = np.asarray(lags, dtype=np.uint64).ravel()

    # Create adamCore instance
    adam_core = _adamCore.adamCore(
        lags=lags,
        E=E,
        T=T,
        S=S,
        nNonSeasonal=int(nNonSeasonal),
        nSeasonal=int(nSeasonal),
        nETS=int(nNonSeasonal + nSeasonal),
        nArima=int(nArima),
        nXreg=int(nXreg),
        constant=bool(constant),
        adamETS=bool(adamETS),
    )

    # Ensure matrices are F-contiguous
    matrixVt = np.asfortranarray(matrixVt, dtype=np.float64)
    matrixWt = np.asfortranarray(matrixWt, dtype=np.float64)
    matrixF = np.asfortranarray(matrixF, dtype=np.float64)
    vectorG = np.asfortranarray(vectorG, dtype=np.float64).ravel()
    indexLookupTable = np.asfortranarray(indexLookupTable, dtype=np.uint64)
    profilesRecent = np.asfortranarray(profilesRecent, dtype=np.float64)
    vectorYt = np.asfortranarray(vectorYt, dtype=np.float64).ravel()
    vectorOt = np.asfortranarray(vectorOt, dtype=np.float64).ravel()

    # Call C++ fit method
    result = adam_core.fit(
        matrixVt=matrixVt,
        matrixWt=matrixWt,
        matrixF=matrixF,
        vectorG=vectorG,
        indexLookupTable=indexLookupTable,
        profilesRecent=profilesRecent,
        vectorYt=vectorYt,
        vectorOt=vectorOt,
        backcast=bool(backcast),
        nIterations=int(nIterations),
        refineHead=bool(refineHead),
        O="n",
    )

    # Convert C++ FitResult to dict for backward compatibility
    return {
        "matVt": np.array(result.states),
        "yFitted": np.array(result.fitted),
        "errors": np.array(result.errors),
        "profile": np.array(result.profile),
    }


def adam_forecaster(
    matrixWt,
    matrixF,
    lags,
    indexLookupTable,
    profilesRecent,
    E,
    T,
    S,
    nNonSeasonal,
    nSeasonal,
    nArima,
    nXreg,
    constant,
    horizon,
):
    """
    Wrapper for adamCore.forecast() method.
    """
    # Convert types
    lags = np.asarray(lags, dtype=np.uint64).ravel()

    # Create adamCore instance
    adam_core = _adamCore.adamCore(
        lags=lags,
        E=E,
        T=T,
        S=S,
        nNonSeasonal=int(nNonSeasonal),
        nSeasonal=int(nSeasonal),
        nETS=int(nNonSeasonal + nSeasonal),
        nArima=int(nArima),
        nXreg=int(nXreg),
        constant=bool(constant),
        adamETS=False,
    )

    # Ensure matrices are F-contiguous
    matrixWt = np.asfortranarray(matrixWt, dtype=np.float64)
    matrixF = np.asfortranarray(matrixF, dtype=np.float64)
    indexLookupTable = np.asfortranarray(indexLookupTable, dtype=np.uint64)
    profilesRecent = np.asfortranarray(profilesRecent, dtype=np.float64)

    # Call C++ forecast method
    result = adam_core.forecast(
        matrixWt=matrixWt,
        matrixF=matrixF,
        indexLookupTable=indexLookupTable,
        profilesRecent=profilesRecent,
        horizon=int(horizon),
    )

    # Return forecast array directly
    return np.array(result.forecast)


def adam_simulator(
    matrixErrors,
    matrixOt,
    arrayVt,
    matrixWt,
    arrayF,
    matrixG,
    lags,
    indexLookupTable,
    profilesRecent,
    E,
    T,
    S,
    nNonSeasonal,
    nSeasonal,
    nArima,
    nXreg,
    constant,
):
    """
    Wrapper for adamCore.simulate() method.
    """
    # Convert types
    lags = np.asarray(lags, dtype=np.uint64).ravel()

    # Create adamCore instance
    adam_core = _adamCore.adamCore(
        lags=lags,
        E=E,
        T=T,
        S=S,
        nNonSeasonal=int(nNonSeasonal),
        nSeasonal=int(nSeasonal),
        nETS=int(nNonSeasonal + nSeasonal),
        nArima=int(nArima),
        nXreg=int(nXreg),
        constant=bool(constant),
        adamETS=False,
    )

    # Ensure matrices are F-contiguous
    matrixErrors = np.asfortranarray(matrixErrors, dtype=np.float64)
    matrixOt = np.asfortranarray(matrixOt, dtype=np.float64)
    arrayVt = np.asfortranarray(arrayVt, dtype=np.float64)
    matrixWt = np.asfortranarray(matrixWt, dtype=np.float64)
    arrayF = np.asfortranarray(arrayF, dtype=np.float64)
    matrixG = np.asfortranarray(matrixG, dtype=np.float64)
    indexLookupTable = np.asfortranarray(indexLookupTable, dtype=np.uint64)
    profilesRecent = np.asfortranarray(profilesRecent, dtype=np.float64)

    # Call C++ simulate method
    result = adam_core.simulate(
        matrixErrors=matrixErrors,
        matrixOt=matrixOt,
        arrayVt=arrayVt,
        matrixWt=matrixWt,
        arrayF=arrayF,
        matrixG=matrixG,
        indexLookupTable=indexLookupTable,
        profilesRecent=profilesRecent,
        E=E,
    )

    # Convert to dict
    return {
        "arrayVt": np.array(result.states),
        "matrixYt": np.array(result.data),
    }
