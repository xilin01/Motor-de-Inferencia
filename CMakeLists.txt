cmake_minimum_required(VERSION 3.20)
project(gpt2 CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# /MT: la biblioteca estándar se enlaza dentro del .exe, que así solo depende de KERNEL32.dll.
set(CMAKE_MSVC_RUNTIME_LIBRARY "MultiThreaded$<$<CONFIG:Debug>:Debug>")
# /utf-8: fuentes y literales en UTF-8 (comentarios y mensajes en español).
add_compile_options(/W4 /utf-8)

# Todo el motor va en una biblioteca estática que comparten la CLI, los tests y el benchmark.
add_library(motor STATIC src/pesos.cpp src/tokenizador.cpp src/operaciones.cpp src/modelo.cpp src/generar.cpp
                         src/cpu.cpp src/lineal_avx2.cpp src/hilos.cpp)
target_include_directories(motor PUBLIC src)
# AVX2 y FMA solo en el archivo de la capa lineal vectorizada. El resto se compila para x64 base
# (SSE2): así la versión ingenua sigue siendo escalar y el binario arranca en cualquier CPU x64.
set_source_files_properties(src/lineal_avx2.cpp PROPERTIES COMPILE_OPTIONS /arch:AVX2)

add_executable(gpt2 src/main.cpp)
target_link_libraries(gpt2 PRIVATE motor)

add_executable(bench bench/bench.cpp)
target_link_libraries(bench PRIVATE motor)

# Tests: un ejecutable por área, lanzados desde la raíz para que encuentren modelos/ y tests/datos/.
enable_testing()
function(prueba nombre)
    add_executable(${nombre} tests/${nombre}.cpp)
    target_include_directories(${nombre} PRIVATE tests)
    target_link_libraries(${nombre} PRIVATE motor)
    add_test(NAME ${nombre} COMMAND ${nombre} WORKING_DIRECTORY ${CMAKE_SOURCE_DIR})
endfunction()

prueba(test_humo)
prueba(test_pesos)
prueba(test_tokenizador)
prueba(test_lineal)
prueba(test_hilos)
prueba(test_logits)
prueba(test_generacion)

# Los mismos tests del modelo para cada variante: camino incremental (KV cache), capa lineal AVX2 y
# varios hilos.
function(variante nombre)
    add_test(NAME ${nombre} COMMAND ${ARGN} WORKING_DIRECTORY ${CMAKE_SOURCE_DIR})
endfunction()
variante(test_logits_incremental      test_logits incremental)
variante(test_logits_avx2             test_logits completo avx2)
variante(test_logits_incremental_avx2 test_logits incremental avx2)
variante(test_generacion_avx2         test_generacion avx2)
variante(test_logits_avx2_hilos             test_logits completo avx2 8)
variante(test_logits_incremental_avx2_hilos test_logits incremental avx2 8)
variante(test_logits_incremental_ingenua_hilos test_logits incremental ingenua 8)
variante(test_generacion_avx2_hilos         test_generacion avx2 8)

# La CLI rechaza un valor numérico a medias ("0,8" no es 0). Sale antes de cargar el modelo.
add_test(NAME cli_valor_invalido COMMAND gpt2 --prompt hi --temperatura 0,8)
set_tests_properties(cli_valor_invalido PROPERTIES PASS_REGULAR_EXPRESSION "valor no num")
