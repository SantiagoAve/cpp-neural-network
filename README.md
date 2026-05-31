# Neural Network Framework in C++
This is a personal proyect focused on learning and practicing what I see on class and Internet. My goal is to create a light, highly optimized framework so I don't have to use PyTorch, since I hate python and I like low-level stuff, but still being able to beat my classmates. This little project should be able to add all the neccesary tools for said goal.

# Concrete Goals
- Improve my C++ programming skills!
- Design from scratch all basic operations and functions of a library like PyTorch, while also adding unique features.
- Start to work with CUDA and GPU programming.
- Train simple models, and compare them using different networks.
- Benchmarking compared to PyTorch; should be able to beat its counterpart.

# Current Roadmap
1. ~~Prepare working environment, tools and knowledge.~~
2. ~~Create the basics; activation, loss, forward.~~
3. ~~Work on the network; layer, network, optimizer, backpropagation.~~
4. Training examples and tests.
5. Expanding the framework; cross-entrophy, more optimizers, convolution, CUDA.
6. Benchamark with PyTorch.

# Compiler Requirements
- Any compiler like GCC or CLang, as long as it supports C++26.
- CMake 3.25 or higher.
- Eigen, for Lineal Algebra operations. "I aint coding allat ;)"

# How to compile
1. Once cloned, you should first write: *git submodule init* and then *git submodule update*. This ensures Eigen will be used.
2. Once done, make a folder for the build, something like: *mkdir build*
3. Then, go inside said folder, you can try: *cd build/*
4. Last but not least, run these commands *in order*, first: *cmake ..*, then *make*, and finally *./main*.

# Anything else?
*Work in progress! Stay tuned :)*
