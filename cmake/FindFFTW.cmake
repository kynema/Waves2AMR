# - Find the FFTW3 library
# This will define the following imported target:
#
#   FFTW::FFTW  - The FFTW library target
#
# and the following variables (for legacy support):
#
#   FFTW_FOUND        - True if FFTW was found
#   FFTW_INCLUDE_DIRS - Include directories for FFTW
#   FFTW_LIBRARIES    - Libraries to link against

# Look for header
find_path(
  FFTW_INCLUDE_DIR
  NAMES fftw3.h
  PATHS
    ${FFTW_DIR}
)

# Look for library (shared or static)
find_library(
  FFTW_LIBRARY
  NAMES fftw3 libfftw3
  PATHS
    ${FFTW_DIR}
)

include(FindPackageHandleStandardArgs)
find_package_handle_standard_args(
  FFTW
  REQUIRED_VARS FFTW_LIBRARY FFTW_INCLUDE_DIR
  VERSION_VAR FFTW_VERSION
)

if(FFTW_FOUND)
  set(FFTW_LIBRARIES ${FFTW_LIBRARY})
  set(FFTW_INCLUDE_DIRS ${FFTW_INCLUDE_DIR})

  if(NOT TARGET FFTW::FFTW)
    add_library(FFTW::FFTW UNKNOWN IMPORTED)
    set_target_properties(FFTW::FFTW PROPERTIES
      IMPORTED_LOCATION "${FFTW_LIBRARY}"
      INTERFACE_INCLUDE_DIRECTORIES "${FFTW_INCLUDE_DIR}"
    )
  endif()
endif()

mark_as_advanced(FFTW_INCLUDE_DIR FFTW_LIBRARY)
