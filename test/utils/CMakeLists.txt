# test/utils/ -- test-support module libraries the suites link. No suites of
# their own, so nothing here registers a ctest entry. Added before the suites
# (test/CMakeLists.txt) so their targets exist when a suite links them.
add_subdirectory(shared_resources)
