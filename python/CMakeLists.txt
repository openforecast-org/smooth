cmake_minimum_required(VERSION 3.16)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED True)
set(CMAKE_CXX_EXTENSIONS OFF)
project(smooth VERSION 1.0.0)

if(SKBUILD)
  message(STATUS "The project is built using scikit-build")
endif()

# Pybind11 (fetch to avoid system dependency)
include(FetchContent)
FetchContent_Declare(
  pybind11
  GIT_REPOSITORY https://github.com/pybind/pybind11.git
  GIT_TAG v2.12.0)
FetchContent_MakeAvailable(pybind11)

# ----------------------------------------------------------------------------
# Armadillo source pre-declaration (overrides carma's default).
#
# carma/cmake/GetArmadillo.cmake declares CarmaArmadillo against
# gitlab.com/conradsnicta/armadillo-code at branch ``15.2.x`` and does a
# full ``git clone`` into the source tree, which is the most failure-prone
# step in the whole build chain: gitlab.com is intermittently unreachable
# from the GitHub Actions manylinux runners, so the populate sub-build
# fails with::
#
#   Build step for carmaarmadillo failed: 1
#   FAILED: carmaarmadillo-populate-prefix/.../carmaarmadillo-populate-download
#
# CMake FetchContent honours first-declared-wins, so this pre-declaration
# is what carma actually consumes.  We swap the ``GIT_REPOSITORY`` +
# ``GIT_TAG`` pair for a ``URL`` pointing at GitLab's branch-archive
# endpoint: same content as the ``15.2.x`` branch carma already targets,
# but the populate is now a single HTTPS GET (no git protocol handshake,
# no checkout, retries handled by CMake's ``URL`` downloader instead of
# git's network stack).  Empirically this is the failure mode that has
# been observed on the runners; the change is invisible to downstream
# consumers because the resulting source tree at the SOURCE_DIR path is
# the same.
#
# If GitLab itself goes down the build still fails, but more cleanly and
# with the URL in the error message.  Pinning to a tagged release
# (``15.2.0``, etc.) is a follow-up — keep the ``15.2.x`` rolling source
# here so behaviour matches carma's existing intent.
FetchContent_Declare(
  CarmaArmadillo
  URL https://gitlab.com/conradsnicta/armadillo-code/-/archive/15.2.x/armadillo-code-15.2.x.tar.gz
  DOWNLOAD_EXTRACT_TIMESTAMP TRUE
  SOURCE_DIR ${CMAKE_CURRENT_SOURCE_DIR}/../src/libs/carma/extern/armadillo-code
)

# CARMA
ADD_SUBDIRECTORY(../src/libs/carma carma)

# ==============================================================================
# BLAS/LAPACK Detection
# ==============================================================================

# BLAS / LAPACK are required by Armadillo's ``eig_gen`` (used by
# ``ar_polynomial_bounds`` for ARIMA reapply and ``eigen_bounds`` for
# admissible bounds), and by ``adamCore`` / ``eigenCalc`` directly
# (which call ``sposv``, ``dgesv``, ``sgeev`` etc. through Armadillo).
# Without LAPACK linked the wheel fails at link time with
# ``LNK2001: unresolved external symbol sposv_``-style errors, or at
# runtime with ``eig_gen(): use of LAPACK must be enabled``.
#
# Detection strategy:
#   1. On Windows, consume the OpenBLAS pre-built release from
#      OpenMathLib (``libopenblas.dll`` ships both BLAS *and* LAPACK
#      in a single library, with no Fortran compiler needed at our
#      build time).  The CI step downloads / extracts the zip and
#      sets ``-DOPENBLAS_HOME=C:/openblas``; we then point
#      ``BLAS_LIBRARIES`` / ``LAPACK_LIBRARIES`` at the MinGW import
#      lib ``libopenblas.dll.a`` (MSVC's linker accepts COFF import
#      libs).  Avoids the whole vcpkg openblas + lapack-reference +
#      vcpkg-gfortran chain that fell apart when
#      ``vcpkg-gfortran@3#3`` switched to LLVMFlang.
#   2. MODULE-mode ``find_package(BLAS)`` / ``find_package(LAPACK)``
#      handles Linux / macOS, where the system packages
#      (openblas-devel + lapack-devel on Linux, brew's openblas on
#      macOS) are found by the canonical search.
#   3. Final fallback only when the dependency is genuinely absent.
if(WIN32)
  if(NOT DEFINED OPENBLAS_HOME AND DEFINED ENV{OPENBLAS_HOME})
    set(OPENBLAS_HOME $ENV{OPENBLAS_HOME})
  endif()
  if(OPENBLAS_HOME)
    # OpenMathLib's pre-built zip lays out as:
    #   <OPENBLAS_HOME>/bin/libopenblas.dll
    #   <OPENBLAS_HOME>/lib/libopenblas.dll.a   (MinGW import lib)
    #   <OPENBLAS_HOME>/include/cblas.h, openblas_config.h, ...
    find_library(OPENBLAS_IMPLIB
                 NAMES libopenblas.dll.a libopenblas.lib openblas
                 PATHS "${OPENBLAS_HOME}/lib"
                 NO_DEFAULT_PATH)
    find_path(OPENBLAS_INCLUDE_DIR
              NAMES openblas_config.h cblas.h
              PATHS "${OPENBLAS_HOME}/include"
                    "${OPENBLAS_HOME}/include/openblas"
              NO_DEFAULT_PATH)
    if(OPENBLAS_IMPLIB AND OPENBLAS_INCLUDE_DIR)
      set(BLAS_FOUND TRUE)
      set(LAPACK_FOUND TRUE)
      set(BLAS_LIBRARIES   ${OPENBLAS_IMPLIB})
      set(LAPACK_LIBRARIES ${OPENBLAS_IMPLIB})
      include_directories(${OPENBLAS_INCLUDE_DIR})
      set(lapackblas_libraries ${OPENBLAS_IMPLIB})
      message(STATUS "Found OpenMathLib OpenBLAS pre-built (BLAS + LAPACK):")
      message(STATUS "  import lib : ${OPENBLAS_IMPLIB}")
      message(STATUS "  headers    : ${OPENBLAS_INCLUDE_DIR}")
    else()
      message(WARNING
        "OPENBLAS_HOME=${OPENBLAS_HOME} set but the import lib / headers "
        "were not found under it.  Falling back to find_package.")
    endif()
  endif()
endif()

if(NOT BLAS_FOUND)
  find_package(BLAS)
  find_package(LAPACK)
  if(LAPACK_FOUND AND BLAS_FOUND)
    set(lapackblas_libraries ${BLAS_LIBRARIES} ${LAPACK_LIBRARIES})
    message(STATUS "BLAS libraries: ${BLAS_LIBRARIES}")
    message(STATUS "LAPACK libraries: ${LAPACK_LIBRARIES}")
  elseif(BLAS_FOUND)
    set(lapackblas_libraries ${BLAS_LIBRARIES})
    add_compile_definitions(ARMA_DONT_USE_LAPACK)
    message(WARNING "LAPACK not found; eig_gen / ar_polynomial_bounds disabled.")
  else()
    message(WARNING "BLAS/LAPACK not found. Disabling BLAS/LAPACK in Armadillo. "
                    "Modules requiring eig_gen / eigvalsh will fail at runtime.")
    add_compile_definitions(ARMA_DONT_USE_BLAS ARMA_DONT_USE_LAPACK)
    set(lapackblas_libraries "")
  endif()
endif()

# Fail loudly on Windows if LAPACK still isn't found — silently
# shipping a wheel without LAPACK leaves every ARIMA / admissible
# code path broken with a runtime crash, which is worse than a
# build failure that surfaces the misconfiguration immediately.
if(WIN32 AND NOT LAPACK_FOUND)
  message(FATAL_ERROR
    "LAPACK not found on Windows. The wheel build expects the OpenMathLib "
    "pre-built OpenBLAS zip extracted to a directory pointed at by the "
    "``OPENBLAS_HOME`` cache variable (or environment variable) — see "
    "``CIBW_BEFORE_ALL_WINDOWS`` in .github/workflows/python_ci.yml.")
endif()

# ==============================================================================
# End BLAS/LAPACK Detection
# ==============================================================================

# Armadillo (use correct CMake package name)
find_package(Armadillo QUIET)
if(Armadillo_FOUND)
  message(STATUS "Found Armadillo")
  include_directories(${ARMADILLO_INCLUDE_DIRS})
else()
  # carma will provide Armadillo headers via FetchContent
  message(STATUS "Armadillo not found - using carma's bundled Armadillo headers")
endif()

# Adam Core - Updated to use new core implementation
pybind11_add_module(_adamCore ../src/python/adamPython.cpp)

# Define PYTHON_BUILD for conditional compilation
target_compile_definitions(_adamCore PRIVATE PYTHON_BUILD)

# CRITICAL: Point to src/ directory so headers can be found
target_include_directories(_adamCore PRIVATE ../src)

target_link_libraries(_adamCore PRIVATE carma::carma)

# Link BLAS/LAPACK if found (not on Windows)
if(lapackblas_libraries)
  target_link_libraries(_adamCore PRIVATE ${lapackblas_libraries})
endif()

if(Armadillo_FOUND)
  if(TARGET Armadillo::armadillo)
    target_link_libraries(_adamCore PRIVATE Armadillo::armadillo)
  else()
    target_link_libraries(_adamCore PRIVATE ${ARMADILLO_LIBRARIES})
  endif()
endif()

# Install to match module name
install(TARGETS _adamCore DESTINATION smooth/adam_general)

# EigenCalc module for eigenvalue calculations
pybind11_add_module(_eigenCalc ../src/python/eigenCalc.cpp)
target_compile_definitions(_eigenCalc PRIVATE PYTHON_BUILD)
target_include_directories(_eigenCalc PRIVATE ../src)
target_link_libraries(_eigenCalc PRIVATE carma::carma)
if(lapackblas_libraries)
  target_link_libraries(_eigenCalc PRIVATE ${lapackblas_libraries})
endif()
if(Armadillo_FOUND)
  if(TARGET Armadillo::armadillo)
    target_link_libraries(_eigenCalc PRIVATE Armadillo::armadillo)
  else()
    target_link_libraries(_eigenCalc PRIVATE ${ARMADILLO_LIBRARIES})
  endif()
endif()
install(TARGETS _eigenCalc DESTINATION smooth/adam_general)

# numDeriv module for numerical derivatives (pracma-exact Hessian)
pybind11_add_module(_numDeriv ../src/python/numDeriv.cpp)
target_compile_definitions(_numDeriv PRIVATE PYTHON_BUILD)
target_include_directories(_numDeriv PRIVATE ../src)
target_link_libraries(_numDeriv PRIVATE carma::carma)
if(lapackblas_libraries)
  target_link_libraries(_numDeriv PRIVATE ${lapackblas_libraries})
endif()
if(Armadillo_FOUND)
  if(TARGET Armadillo::armadillo)
    target_link_libraries(_numDeriv PRIVATE Armadillo::armadillo)
  else()
    target_link_libraries(_numDeriv PRIVATE ${ARMADILLO_LIBRARIES})
  endif()
endif()
install(TARGETS _numDeriv DESTINATION smooth/adam_general)

# OLS module - shared least-squares backend (msdecompose parity with R)
pybind11_add_module(_ols ../src/python/olsWrap.cpp)
target_compile_definitions(_ols PRIVATE PYTHON_BUILD)
target_include_directories(_ols PRIVATE ../src)
target_link_libraries(_ols PRIVATE carma::carma)
if(lapackblas_libraries)
  target_link_libraries(_ols PRIVATE ${lapackblas_libraries})
endif()
if(Armadillo_FOUND)
  if(TARGET Armadillo::armadillo)
    target_link_libraries(_ols PRIVATE Armadillo::armadillo)
  else()
    target_link_libraries(_ols PRIVATE ${ARMADILLO_LIBRARIES})
  endif()
endif()
install(TARGETS _ols DESTINATION smooth/adam_general)
