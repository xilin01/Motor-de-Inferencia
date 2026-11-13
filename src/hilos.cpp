#include "hilos.hpp"

#include <algorithm>

Hilos::Hilos(int n) {
    for (int i = 1; i < n; ++i) hilos_.emplace_back([this] { trabajador(); });
}

Hilos::~Hilos() {
    {
        std::lock_guard lock(mutex_);
        salir_ = true;
    }
    hay_trabajo_.notify_all();
    for (std::thread& h : hilos_) h.join();
}

void Hilos::paralelo(size_t total, size_t trozo, const std::function<void(size_t, size_t)>& tarea) {
    {
        std::lock_guard lock(mutex_);
        tarea_ = &tarea;
        total_ = total;
        trozo_ = std::max<size_t>(trozo, 1);
        siguiente_ = 0;
        pendientes_ = static_cast<int>(hilos_.size());
        ++ronda_;
    }
    hay_trabajo_.notify_all();

    consumir();  // el hilo que llama también trabaja

    std::unique_lock lock(mutex_);
    terminado_.wait(lock, [this] { return pendientes_ == 0; });
}

void Hilos::trabajador() {
    uint64_t vista = 0;  // última ronda atendida
    std::unique_lock lock(mutex_);
    while (true) {
        hay_trabajo_.wait(lock, [&] { return salir_ || ronda_ != vista; });
        if (salir_) return;
        vista = ronda_;

        lock.unlock();
        consumir();
        lock.lock();

        if (--pendientes_ == 0) terminado_.notify_one();
    }
}

// Coge trozos hasta que no queden. fetch_add es atómico: dos hilos nunca reciben el mismo trozo.
void Hilos::consumir() {
    while (true) {
        const size_t ini = siguiente_.fetch_add(trozo_);
        if (ini >= total_) return;
        (*tarea_)(ini, std::min(ini + trozo_, total_));
    }
}
