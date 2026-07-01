cmake_minimum_required(VERSION 3.14)
project(llm_tui_monitor CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# --- Download FTXUI automatically ---
include(FetchContent)
FetchContent_Declare(
  ftxui
  GIT_REPOSITORY https://github.com/ArthurSonzogni/FTXUI.git
  GIT_TAG        v5.0.0
)
FetchContent_MakeAvailable(ftxui)

# --- llama.cpp paths ---
set(LLAMA_DIR ${CMAKE_SOURCE_DIR}/llama.cpp)
set(LLAMA_LIB_DIR ${LLAMA_DIR}/build/bin)

# --- Your main program ---
add_executable(llm_monitor
  src/main.cpp
  src/capture_layer.cpp
  src/tui_dashboard.cpp
)

# --- Include paths (where header files live) ---
target_include_directories(llm_monitor PRIVATE
  include/
  ${LLAMA_DIR}/include
  ${LLAMA_DIR}/ggml/include
  ${LLAMA_DIR}/common
)

# --- Link against the .so files ---
target_link_libraries(llm_monitor PRIVATE
  ftxui::screen
  ftxui::dom
  ftxui::component
  ${LLAMA_LIB_DIR}/libllama.so
  ${LLAMA_LIB_DIR}/libggml.so
  ${LLAMA_LIB_DIR}/libggml-base.so
  ${LLAMA_LIB_DIR}/libggml-cpu.so
  pthread
  dl
)

# --- Tell the executable WHERE to find .so files at runtime ---
set_target_properties(llm_monitor PROPERTIES
  BUILD_RPATH ${LLAMA_LIB_DIR}
  INSTALL_RPATH ${LLAMA_LIB_DIR}
)
