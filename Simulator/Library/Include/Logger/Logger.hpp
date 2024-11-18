/**
 * @brief Todo
 * @date 2024/10/28
 */

#ifndef VT_PHYSICS_LOGGER_HPP
#define VT_PHYSICS_LOGGER_HPP

#include "spdlog/spdlog.h"
#include "spdlog/sinks/basic_file_sink.h"

#include <iostream>

namespace VT_Physics {

#define LOG_INFO(msg) \
    spdlog::info(msg);

#define LOG_ERROR(msg) \
    spdlog::error(msg);

#ifdef NDEBUG

#define LOG_WARNING(msg) {};

#else

#define LOG_WARNING(msg) \
    spdlog::warn(msg);

#endif

#define SYS_PAUSE() \
    LOG_INFO("Simulation done! \n\n Press 'Enter' to exit..."); \
    std::cin.get();

}

#endif //VT_PHYSICS_LOGGER_HPP
