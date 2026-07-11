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
    refineHead=True,
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
        nComponents=int(
            nNonSeasonal + nSeasonal + nArima + nXreg + int(bool(constant))
        ),
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
        refineHead=bool(refineHead),
    )

    # Convert to dict
    return {
        "arrayVt": np.array(result.states),
        "matrixYt": np.array(result.data),
    }


def adam_reapply(
    matrixYt,
    matrixOt,
    arrayVt,
    arrayWt,
    arrayF,
    matrixG,
    lags,
    indexLookupTable,
    arrayProfilesRecent,
    E,
    T,
    S,
    nNonSeasonal,
    nSeasonal,
    nArima,
    nXreg,
    constant,
    adamETS,
    backcast,
):
    """Wrapper for ``adamCore.reapply()`` — re-runs the in-sample ADAM kernel
    for ``nsim`` parameter draws and returns the per-draw states, fitted and
    final profile.

    See ``ADAM.reapply`` for the high-level API.
    """
    lags = np.asarray(lags, dtype=np.uint64).ravel()

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
        nComponents=int(
            nNonSeasonal + nSeasonal + nArima + nXreg + int(bool(constant))
        ),
        constant=bool(constant),
        adamETS=bool(adamETS),
    )

    matrixYt = np.asfortranarray(matrixYt, dtype=np.float64)
    matrixOt = np.asfortranarray(matrixOt, dtype=np.float64)
    arrayVt = np.asfortranarray(arrayVt, dtype=np.float64)
    arrayWt = np.asfortranarray(arrayWt, dtype=np.float64)
    arrayF = np.asfortranarray(arrayF, dtype=np.float64)
    matrixG = np.asfortranarray(matrixG, dtype=np.float64)
    indexLookupTable = np.asfortranarray(indexLookupTable, dtype=np.uint64)
    arrayProfilesRecent = np.asfortranarray(arrayProfilesRecent, dtype=np.float64)

    result = adam_core.reapply(
        matrixYt=matrixYt,
        matrixOt=matrixOt,
        arrayVt=arrayVt,
        arrayWt=arrayWt,
        arrayF=arrayF,
        matrixG=matrixG,
        indexLookupTable=indexLookupTable,
        arrayProfilesRecent=arrayProfilesRecent,
        backcast=bool(backcast),
    )

    return {
        "states": np.array(result.states),
        "fitted": np.array(result.fitted),
        "profile": np.array(result.profile),
    }


def adam_reforecast(
    arrayErrors,
    arrayOt,
    arrayWt,
    arrayF,
    matrixG,
    lags,
    indexLookupTable,
    arrayProfileRecent,
    E,
    T,
    S,
    nNonSeasonal,
    nSeasonal,
    nArima,
    nXreg,
    constant,
    adamETS=False,
):
    """Wrapper for ``adamCore.reforecast()`` — generates ``(h, nsim, nsim)``
    forecast paths from per-replicate state matrices and per-replicate error
    samples. See ``ADAM.reforecast`` for the high-level API.
    """
    lags = np.asarray(lags, dtype=np.uint64).ravel()

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
        nComponents=int(
            nNonSeasonal + nSeasonal + nArima + nXreg + int(bool(constant))
        ),
        constant=bool(constant),
        adamETS=bool(adamETS),
    )

    arrayErrors = np.asfortranarray(arrayErrors, dtype=np.float64)
    arrayOt = np.asfortranarray(arrayOt, dtype=np.float64)
    arrayWt = np.asfortranarray(arrayWt, dtype=np.float64)
    arrayF = np.asfortranarray(arrayF, dtype=np.float64)
    matrixG = np.asfortranarray(matrixG, dtype=np.float64)
    indexLookupTable = np.asfortranarray(indexLookupTable, dtype=np.uint64)
    arrayProfileRecent = np.asfortranarray(arrayProfileRecent, dtype=np.float64)

    result = adam_core.reforecast(
        arrayErrors=arrayErrors,
        arrayOt=arrayOt,
        arrayWt=arrayWt,
        arrayF=arrayF,
        matrixG=matrixG,
        indexLookupTable=indexLookupTable,
        arrayProfileRecent=arrayProfileRecent,
        E=E,
    )

    return {"data": np.array(result.data)}
