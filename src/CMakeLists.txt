# Sources
set(SOURCES
    neon/neon_kernels.cpp
    vulkan/vulkan_backend.cpp
)

if(ENABLE_VULKAN)
    # Shader compilation logic
    find_program(GLSLANG_VALIDATOR glslangValidator)
    if(GLSLANG_VALIDATOR)
        # Define shaders to compile
        set(SHADERS
            vec_add.comp.glsl
            vec_mul.comp.glsl
            vec_dot.comp.glsl
            conv1d.comp.glsl
        )

        set(SPV_OUTPUTS "")

        # Create the directory for the output file
        file(MAKE_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR})

        foreach(SHADER ${SHADERS})
             set(SOURCE_FILE ${CMAKE_CURRENT_SOURCE_DIR}/vulkan/shaders/${SHADER})
             set(OUTPUT_FILE ${CMAKE_CURRENT_BINARY_DIR}/${SHADER})
             string(REPLACE ".glsl" ".spv" OUTPUT_FILE ${OUTPUT_FILE}) # e.g. vec_add.comp.spv

             list(APPEND SPV_OUTPUTS ${OUTPUT_FILE})

             add_custom_command(
                OUTPUT ${OUTPUT_FILE}
                COMMAND ${GLSLANG_VALIDATOR} -V ${SOURCE_FILE} -o ${OUTPUT_FILE}
                DEPENDS ${SOURCE_FILE}
                COMMENT "Compiling ${SHADER} to SPIR-V"
            )
        endforeach()

        # We create a custom target to ensure shaders are built.
        add_custom_target(optmath_shaders ALL DEPENDS ${SPV_OUTPUTS})

        # Install shaders to a known location (datadir/optmathkernels/shaders)
        # And also copy them to the binary directory of the example app for easy finding
        install(FILES ${SPV_OUTPUTS} DESTINATION ${CMAKE_INSTALL_DATADIR}/optmathkernels/shaders)

    else()
        message(WARNING "glslangValidator not found. Shaders will not be compiled.")
    endif()
endif()

add_library(OptMathKernels ${SOURCES})
add_library(OptMathKernels::OptMathKernels ALIAS OptMathKernels)

# Include directories
target_include_directories(OptMathKernels
    PUBLIC
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/../include>
        $<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}>
)

# Link dependencies
target_link_libraries(OptMathKernels PUBLIC Eigen3::Eigen)

if(ENABLE_VULKAN)
    target_link_libraries(OptMathKernels PRIVATE Vulkan::Vulkan)
    if(GLSLANG_VALIDATOR)
        add_dependencies(OptMathKernels optmath_shaders)
    endif()
endif()

# Installation
install(TARGETS OptMathKernels
    EXPORT  OptMathKernelsTargets
    ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR}
    LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
    RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR}
)

install(DIRECTORY ../include/
    DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}
)

# Export configuration
install(EXPORT OptMathKernelsTargets
    FILE OptMathKernelsTargets.cmake
    NAMESPACE OptMathKernels::
    DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/OptMathKernels
)

include(CMakePackageConfigHelpers)
configure_package_config_file(
    ${CMAKE_CURRENT_SOURCE_DIR}/../cmake/OptMathKernelsConfig.cmake.in
    ${CMAKE_CURRENT_BINARY_DIR}/OptMathKernelsConfig.cmake
    INSTALL_DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/OptMathKernels
)

install(FILES
    ${CMAKE_CURRENT_BINARY_DIR}/OptMathKernelsConfig.cmake
    DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/OptMathKernels
)
