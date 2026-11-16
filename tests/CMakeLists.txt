# TESTS ------------------------------------------------------------------
add_executable(nn_test_activations test_activations_loss.cpp)
add_executable(nn_test_gradients    test_gradient_check.cpp)
add_executable(nn_test_training     test_training.cpp)

foreach(target nn_test_activations nn_test_gradients nn_test_training)
    target_link_libraries(${target} PRIVATE neural)
    nn_apply_warnings(${target})
    nn_apply_sanitizers(${target})
endforeach()
# ------------------------------------------------------------------------
