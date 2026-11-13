cmake_minimum_required(VERSION 3.20)
project(gpt2 CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# /MT: la biblioteca estándar se enlaza dentro del .exe, que así solo depende de KERNEL32.dll.
set(CMAKE_MSVC_RUNTIME_LIBRARY "MultiThreaded$<$<CONFIG:Debug>:Debug>")
# /utf-8: fuentes y literales en UTF-8 (comentarios y mensajes en español).
add_compile_options(/W4 /utf-8)

add_executable(gpt2 src/main.cpp)

# Tests: un ejecutable por área, lanzados desde la raíz para que encuentren modelos/ y tests/datos/.
enable_testing()
function(prueba nombre)
    add_executable(${nombre} tests/${nombre}.cpp)
    target_include_directories(${nombre} PRIVATE src tests)
    add_test(NAME ${nombre} COMMAND ${nombre} WORKING_DIRECTORY ${CMAKE_SOURCE_DIR})
endfunction()

prueba(test_humo)
