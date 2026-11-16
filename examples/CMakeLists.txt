# XOR Demo --------------------------------------
add_executable(xor_demo XorDemo.cpp)
target_link_libraries(xor_demo PRIVATE neural)

nn_apply_warnings(xor_demo)
nn_apply_sanitizers(xor_demo)
# -----------------------------------------------
