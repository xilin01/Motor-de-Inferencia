// Grupo de hilos fijo para repartir un bucle. Los hilos se crean una vez y esperan dormidos entre
// llamadas: crear un hilo por cada multiplicación de matrices costaría más que la multiplicación.
#pragma once
#include <atomic>
#include <condition_variable>
#include <cstddef>
#include <cstdint>
#include <functional>
#include <mutex>
#include <thread>
#include <vector>

class Hilos {
public:
    // n hilos en total, contando el que llama a paralelo(): se crean n - 1.
    explicit Hilos(int n);
    ~Hilos();
    Hilos(const Hilos&) = delete;
    Hilos& operator=(const Hilos&) = delete;

    int n() const { return static_cast<int>(hilos_.size()) + 1; }

    // Ejecuta tarea(ini, fin) sobre trozos disjuntos de tamaño `trozo` que cubren [0, total) y
    // vuelve cuando todos han terminado. Los trozos se reparten sobre la marcha: el hilo que
    // acaba uno coge el siguiente. La tarea no debe lanzar excepciones.
    void paralelo(size_t total, size_t trozo, const std::function<void(size_t, size_t)>& tarea);

private:
    void trabajador();
    void consumir();

    std::vector<std::thread> hilos_;
    std::mutex mutex_;
    std::condition_variable hay_trabajo_, terminado_;

    // Trabajo en curso. Lo escribe paralelo() con el mutex cogido antes de despertar a nadie.
    const std::function<void(size_t, size_t)>* tarea_ = nullptr;
    size_t total_ = 0, trozo_ = 1;
    std::atomic<size_t> siguiente_{0};  // primer índice aún sin repartir

    uint64_t ronda_ = 0;  // sube en cada paralelo(): así un trabajador distingue un trabajo nuevo
    int pendientes_ = 0;  // trabajadores que aún no han acabado la ronda actual
    bool salir_ = false;
};
