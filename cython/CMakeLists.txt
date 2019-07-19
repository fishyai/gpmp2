# Examples
message(STATUS "Installing Cython toolbox scripts")
message("INSTALL PATH: ${GPMP2_CYTHON_INSTALL_PATH}")

# This will likely be broken by future versions of GTSAM but it's a hack because their script
# doesn't seem to make __init__ files for us
file(TOUCH "${CMAKE_CURRENT_BINARY_DIR}/gpmp2/__init__.py")

file(MAKE_DIRECTORY "${CMAKE_CURRENT_BINARY_DIR}/gpmp2/util")
file(TOUCH "${CMAKE_CURRENT_BINARY_DIR}/gpmp2/util/__init__.py")

install(FILES "${CMAKE_CURRENT_BINARY_DIR}/gpmp2/__init__.py" DESTINATION ${GPMP2_CYTHON_INSTALL_PATH})
install(DIRECTORY "${CMAKE_CURRENT_BINARY_DIR}/gpmp2/util" DESTINATION ${GPMP2_CYTHON_INSTALL_PATH})
install(FILES "${CMAKE_CURRENT_BINARY_DIR}/gpmp2/util/__init__.py" DESTINATION "${GPMP2_CYTHON_INSTALL_PATH}/util")
install_cython_scripts("${CMAKE_CURRENT_SOURCE_DIR}/cython" "${GPMP2_CYTHON_INSTALL_PATH}/util" "*.py")
