target_sources(${w2a_utests_name}
               PRIVATE
               test_readops.cpp
               test_ascii.cpp
               test_fftexamples.cpp
               test_hosgrid.cpp
               test_combined.cpp
               test_combined_NWT.cpp
               test_AMReXexamples.cpp
               test_interp_to_mfab.cpp
               utest_main.cpp)
target_include_directories(${w2a_utests_name} PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})
target_link_libraries(${w2a_utests_name} PRIVATE gtest)
target_link_libraries(${w2a_utests_name} PUBLIC ${w2a_lib_name})
target_include_directories(${w2a_utests_name} SYSTEM PRIVATE
  ${CMAKE_SOURCE_DIR}/submods/googletest/googletest/include)

add_executable(Waves2AMR_padding_tests test_ocean_padding.cpp test_velocity_stream.cpp test_ocean_reconstruction.cpp test_nwt_reconstruction.cpp test_cubic_interpolation.cpp test_interp_to_mfab.cpp test_modes_io.cpp test_ascii.cpp test_readops.cpp utest_main.cpp)
target_link_libraries(Waves2AMR_padding_tests PRIVATE gtest ${w2a_lib_name})
set_cuda_build_properties(Waves2AMR_padding_tests)
add_test(NAME ocean_padding COMMAND Waves2AMR_padding_tests --gtest_filter=OceanPadding.*)
add_test(NAME velocity_stream COMMAND Waves2AMR_padding_tests --gtest_filter=VelocityStream.*)
add_test(NAME ocean_reconstruction COMMAND Waves2AMR_padding_tests --gtest_filter=OceanReconstruction.*)
add_test(NAME cubic_interpolation COMMAND Waves2AMR_padding_tests --gtest_filter=CubicInterpolation.*)
add_test(NAME interpolation COMMAND Waves2AMR_padding_tests --gtest_filter=InterpToMFabTest.*)
add_test(NAME nwt_reconstruction COMMAND Waves2AMR_padding_tests --gtest_filter=NwtReconstruction.*)
add_test(NAME modes_io COMMAND Waves2AMR_padding_tests --gtest_filter=ModesIO.*:AsciiReadTest.*:ReadOpsTest.*)
set_tests_properties(modes_io PROPERTIES WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR})
